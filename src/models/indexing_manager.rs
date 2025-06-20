use super::{
    buffered_io::BufIoError,
    collection::{Collection, RawVectorEmbedding},
    collection_transaction::{
        BackgroundExplicitTransaction, ImplicitTransaction, Progress, Summary, TransactionStatus,
    },
    common::WaCustomError,
    meta_persist::update_background_version,
    types::VectorId,
    versioning::VersionNumber,
    wal::{VectorOp, WALFile},
};
use crate::config_loader::{Config, VectorsIndexingMode};
use chrono::Utc;
use parking_lot::RwLock;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    ThreadPool,
};
use std::{
    fs,
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
};

pub struct IndexingManager {
    thread: Option<JoinHandle<()>>,
    channel: mpsc::Sender<VersionNumber>,
}

impl IndexingManager {
    pub fn new(
        collection: Arc<Collection>,
        config: Arc<Config>,
        threadpool: Arc<ThreadPool>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel::<VersionNumber>();

        let thread = thread::spawn(move || {
            for version_hash in receiver {
                Self::index_version(&collection, &config, &threadpool, version_hash).unwrap();
            }
        });

        Self {
            thread: Some(thread),
            channel: sender,
        }
    }

    pub fn trigger(&self, version: VersionNumber) {
        self.channel.send(version).unwrap()
    }

    pub fn index_version(
        collection: &Collection,
        config: &Config,
        threadpool: &ThreadPool,
        version: VersionNumber,
    ) -> Result<(), WaCustomError> {
        let txn = BackgroundExplicitTransaction::from_version_id_and_number(collection, version);
        let wal = WALFile::from_existing(&collection.get_path(), version)?;
        let vectors_count = wal.vectors_count();
        let status = collection
            .transaction_status_map
            .get_latest(&version)
            .unwrap();
        let start = Utc::now();
        *status.write() = TransactionStatus::InProgress {
            started_at: start,
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
        let errors = RwLock::new(Vec::new());
        threadpool.scope(|s| {
            while let Some(op) = wal.read()? {
                s.spawn(|_| {
                    let fallible = || match op {
                        VectorOp::Upsert(embeddings) => {
                            let len = embeddings.len() as u32;
                            match config.indexing.mode {
                                VectorsIndexingMode::Sequential => {
                                    collection.index_embeddings(embeddings, txn.version, config)?;
                                }
                                VectorsIndexingMode::Batch { batch_size } => {
                                    embeddings.into_par_iter().chunks(batch_size).try_for_each(
                                        |embeddings| {
                                            collection.index_embeddings(
                                                embeddings,
                                                txn.version,
                                                config,
                                            )
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
                                started_at: start,
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
                        }
                        VectorOp::Delete(_vector_id) => unimplemented!(),
                    };

                    if let Err(err) = fallible() {
                        errors.write().push(err);
                    }
                });
            }
            Ok::<_, WaCustomError>(())
        })?;
        let errors = errors.into_inner();
        if let Some(err) = errors.into_iter().next() {
            return Err(err);
        }
        let end = Utc::now();
        let delta = end - start;
        let delta_seconds = (delta.num_seconds() as u32).max(1);
        let total_records_indexed = records_indexed.load(Ordering::Relaxed);
        *status.write() = TransactionStatus::Complete {
            started_at: start,
            summary: Summary {
                total_records_indexed,
                duration_seconds: delta_seconds,
                average_rate_per_second: total_records_indexed as f32 / delta_seconds as f32,
            },
            last_updated: end,
        };
        txn.pre_commit(collection, config)?;
        update_background_version(&collection.lmdb, version)?;
        fs::remove_file(collection.get_path().join(format!("{}.wal", *version)))
            .map_err(BufIoError::Io)
            .unwrap();
        Ok(())
    }

    pub fn implicit_txn_upsert(
        collection: &Collection,
        transaction: &ImplicitTransaction,
        config: &Config,
        embeddings: Vec<RawVectorEmbedding>,
    ) -> Result<(), WaCustomError> {
        let vectors_count = embeddings.len() as u32;
        let version = transaction.version(collection)?;
        transaction.append_to_wal(collection, VectorOp::Upsert(embeddings.clone()))?;
        let status = collection
            .transaction_status_map
            .get_latest(&version)
            .unwrap();
        match config.indexing.mode {
            VectorsIndexingMode::Sequential => {
                collection.index_embeddings(embeddings, version, config)?;
            }
            VectorsIndexingMode::Batch { batch_size } => {
                embeddings
                    .into_par_iter()
                    .chunks(batch_size)
                    .try_for_each(|embeddings| {
                        collection.index_embeddings(embeddings, version, config)
                    })?;
            }
        }
        let mut status = status.write();
        status.increment_vector_count(vectors_count);
        status.update_last_updated();
        Ok(())
    }

    pub fn implicit_txn_delete(
        _collection: &Collection,
        _transaction: &ImplicitTransaction,
        _config: &Config,
        _vector_id: VectorId,
    ) -> Result<(), WaCustomError> {
        unimplemented!()
    }
}

impl Drop for IndexingManager {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
