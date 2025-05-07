use super::{
    buffered_io::BufIoError,
    collection::Collection,
    collection_transaction::{
        BackgroundCollectionTransaction, Progress, Summary, TransactionStatus,
    },
    common::WaCustomError,
    meta_persist::update_background_version,
    versioning::VersionHash,
    wal::{VectorOp, WALFile},
};
use crate::config_loader::{Config, VectorsIndexingMode};
use chrono::Utc;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{
    fs,
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
};

pub struct IndexingManager {
    thread: Option<JoinHandle<Result<(), WaCustomError>>>,
    channel: mpsc::Sender<VersionHash>,
}

impl IndexingManager {
    pub fn new(collection: Arc<Collection>, config: Config) -> Self {
        let (sender, receiver) = mpsc::channel();

        let thread = thread::spawn(move || {
            for version_hash in receiver {
                Self::index_version(&collection, &config, version_hash)?;
            }
            Ok(())
        });

        Self {
            thread: Some(thread),
            channel: sender,
        }
    }

    pub fn trigger(&self, version_hash: VersionHash) {
        self.channel.send(version_hash).unwrap()
    }

    pub fn index_version(
        collection: &Collection,
        config: &Config,
        version_hash: VersionHash,
    ) -> Result<(), WaCustomError> {
        let txn = BackgroundCollectionTransaction::from_version_id_and_number(
            collection,
            version_hash,
            version_hash.version_number(),
        );
        let wal = WALFile::new(&collection.get_path(), version_hash)?;
        let vectors_count = wal.vectors_count();
        let status = collection
            .transaction_status_map
            .get_latest(&version_hash)
            .unwrap();
        let start = Utc::now();
        *status.write() = TransactionStatus::InProgress {
            progress: Progress {
                percentage_done: 0.0,
                records_indexed: 0,
                total_records: vectors_count,
                rate_per_second: 0.0,
                estimated_time_remaining_seconds: u32::MAX,
            },
            last_updated: start,
        };
        let records_indexed = AtomicU32::new(0);
        thread::scope(|s| {
            let mut handles = Vec::new();
            while let Some(op) = wal.read()? {
                match op {
                    VectorOp::Upsert(embeddings) => {
                        let handle = s.spawn(|| {
                            let len = embeddings.len() as u32;
                            match config.indexing.mode {
                                VectorsIndexingMode::Sequential => {
                                    collection.index_embeddings(embeddings, &txn, config)?;
                                }
                                VectorsIndexingMode::Batch { batch_size } => {
                                    embeddings.into_par_iter().chunks(batch_size).try_for_each(
                                        |embeddings| {
                                            collection.index_embeddings(embeddings, &txn, config)
                                        },
                                    )?;
                                }
                            }
                            let old_count = records_indexed.fetch_add(len, Ordering::AcqRel);
                            let new_count = old_count + len;
                            let now = Utc::now();
                            let delta = now - start;
                            let delta_seconds = (delta.num_seconds() as u32).max(1);
                            let rate_per_second = new_count as f32 / delta_seconds as f32;
                            let mut status = status.write();
                            *status = TransactionStatus::InProgress {
                                progress: Progress {
                                    percentage_done: (new_count as f32 / vectors_count as f32)
                                        * 100.0,
                                    records_indexed: new_count,
                                    total_records: vectors_count,
                                    rate_per_second,
                                    estimated_time_remaining_seconds: (vectors_count
                                        .saturating_sub(new_count)
                                        as f32
                                        / rate_per_second)
                                        .ceil()
                                        as u32,
                                },
                                last_updated: now,
                            };
                            Ok::<_, WaCustomError>(())
                        });
                        handles.push(handle);
                    }
                    VectorOp::Delete(_vector_id) => unimplemented!(),
                }
            }
            for handle in handles {
                handle.join().unwrap()?;
            }
            Ok::<_, WaCustomError>(())
        })?;
        let end = Utc::now();
        let delta = end - start;
        let delta_seconds = (delta.num_seconds() as u32).max(1);
        txn.pre_commit(collection, config)?;
        update_background_version(&collection.lmdb, version_hash)?;
        fs::remove_file(collection.get_path().join(format!("{}.wal", *version_hash)))
            .map_err(BufIoError::Io)?;
        let total_records_indexed = records_indexed.load(Ordering::Relaxed);
        *status.write() = TransactionStatus::Complete {
            summary: Summary {
                total_records_indexed,
                duration_seconds: delta_seconds,
                average_rate_per_second: total_records_indexed as f32 / delta_seconds as f32,
            },
            last_updated: end,
        };
        Ok(())
    }
}

impl Drop for IndexingManager {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
