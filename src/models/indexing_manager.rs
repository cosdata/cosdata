use super::{
    buffered_io::BufIoError,
    collection::{Collection, RawVectorEmbedding},
    collection_transaction::{
        BackgroundExplicitTransaction, ExplicitTransactionID, ImplicitTransaction, ProcessingStats,
        TransactionStatus,
    },
    common::WaCustomError,
    meta_persist::update_background_version,
    types::VectorId,
    versioning::{VersionNumber, VersionSource},
    wal::{VectorOp, WALFile},
};
use crate::config_loader::{Config, VectorsIndexingMode};
use chrono::{Duration, Utc};
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
    channel: mpsc::Sender<(ExplicitTransactionID, VersionNumber)>,
}

impl IndexingManager {
    pub fn new(
        collection: Arc<Collection>,
        config: Arc<Config>,
        threadpool: Arc<ThreadPool>,
    ) -> Self {
        let (sender, receiver) = mpsc::channel::<(ExplicitTransactionID, VersionNumber)>();

        let thread = thread::spawn(move || {
            for (txn_id, version_hash) in receiver {
                Self::index_explicit_txn(&collection, &config, &threadpool, txn_id, version_hash)
                    .unwrap();
            }
        });

        Self {
            thread: Some(thread),
            channel: sender,
        }
    }

    pub fn trigger(&self, txn_id: ExplicitTransactionID, version: VersionNumber) {
        self.channel.send((txn_id, version)).unwrap()
    }

    pub fn index_explicit_txn(
        collection: &Collection,
        config: &Config,
        threadpool: &ThreadPool,
        txn_id: ExplicitTransactionID,
        version: VersionNumber,
    ) -> Result<(), WaCustomError> {
        collection.is_indexing.store(true, Ordering::Relaxed);
        let txn = BackgroundExplicitTransaction::from_version_id_and_number(collection, version);
        let wal = WALFile::from_existing(&collection.get_path(), version)?;
        let total_records_upserted = wal.records_upserted();
        let total_operations = wal.total_operations();
        let status = collection
            .transaction_status_map
            .get_latest(&txn_id)
            .unwrap();
        let start = Utc::now();
        *status.write() = TransactionStatus::InProgress {
            started_at: start,
            stats: ProcessingStats {
                records_upserted: 0,
                records_deleted: 0,
                total_operations,
                percentage_complete: 0.0,
                processing_time_seconds: None,
                average_throughput: None,
                current_processing_rate: Some(1.0),
                estimated_completion: None,
                version_created: Some(version),
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
                            let old_count = records_indexed.fetch_add(len, Ordering::Acquire);
                            let new_count = old_count + len;
                            let now = Utc::now();
                            let delta = now - start;
                            let delta_seconds = (delta.num_seconds() as u32).max(1);
                            let rate_per_second = new_count as f32 / delta_seconds as f32;
                            let mut status = status.write();
                            *status = TransactionStatus::InProgress {
                                started_at: start,
                                stats: ProcessingStats {
                                    records_upserted: new_count,
                                    records_deleted: 0,
                                    total_operations,
                                    percentage_complete: (new_count as f32
                                        / total_records_upserted as f32)
                                        * 100.0,
                                    processing_time_seconds: None,
                                    average_throughput: None,
                                    current_processing_rate: Some(rate_per_second),
                                    estimated_completion: Some(
                                        Utc::now()
                                            + Duration::seconds(
                                                ((total_records_upserted - new_count) as f32
                                                    / rate_per_second)
                                                    as i64,
                                            ),
                                    ),
                                    version_created: Some(version),
                                },
                                last_updated: now,
                            };

                            Ok::<_, WaCustomError>(())
                        }
                        VectorOp::Delete(vector_id) => {
                            log::info!("Processing delete operation for vector_id: {}", vector_id);
                            
                            let old_count = records_indexed.load(Ordering::Acquire);
                            let now = Utc::now();
                            let delta = now - start;
                            let delta_seconds = (delta.num_seconds() as u32).max(1);
                            let rate_per_second = old_count as f32 / delta_seconds as f32;
                            let mut status = status.write();
                            *status = TransactionStatus::InProgress {
                                started_at: start,
                                stats: ProcessingStats {
                                    records_upserted: old_count,
                                    records_deleted: 1,
                                    total_operations,
                                    percentage_complete: (old_count as f32
                                        / total_records_upserted as f32)
                                        * 100.0,
                                    processing_time_seconds: None,
                                    average_throughput: None,
                                    current_processing_rate: Some(rate_per_second),
                                    estimated_completion: Some(
                                        Utc::now()
                                            + Duration::seconds(
                                                ((total_records_upserted - old_count) as f32
                                                    / rate_per_second)
                                                    as i64,
                                            ),
                                    ),
                                    version_created: Some(version),
                                },
                                last_updated: now,
                            };

                            Ok::<_, WaCustomError>(())
                        }
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
        status.write().complete(version);
        txn.pre_commit(collection, config)?;
        update_background_version(&collection.lmdb, version)?;
        fs::remove_file(collection.get_path().join(format!("{}.wal", *version)))
            .map_err(BufIoError::Io)
            .unwrap();
        collection.is_indexing.store(false, Ordering::Relaxed);
        // --- FIX: Update version metadata for explicit transactions ---
        use std::collections::HashSet;
        let mut seen_ids = HashSet::new();
        for (_hash, arc_quotient) in collection.external_to_internal_map.root.quotients.map.to_list() {
            let internal_id = arc_quotient.value.read().value;
            seen_ids.insert(internal_id);
        }
        let vector_count = seen_ids.len() as u32;
        collection.vcs.update_version_metadata(
            version,
            vector_count,
            0, // records_deleted
            vector_count, // total_operations (approximate)
        ).map_err(|e| WaCustomError::DatabaseError(format!("Failed to update version metadata: {e}")))?;
        Ok(())
    }

    fn index_implicit_txn(
        collection: &Collection,
        config: &Config,
        threadpool: &ThreadPool,
        version: VersionNumber,
    ) -> Result<(), WaCustomError> {
        collection.is_indexing.store(true, Ordering::Relaxed);
        let txn = BackgroundExplicitTransaction::from_version_id_and_number(collection, version);
        let wal = WALFile::from_existing(&collection.get_path(), version)?;
        let errors = RwLock::new(Vec::new());
        threadpool.scope(|s| {
            while let Some(op) = wal.read()? {
                s.spawn(|_| {
                    let fallible = || match op {
                        VectorOp::Upsert(embeddings) => {
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

                            Ok::<_, WaCustomError>(())
                        }
                        VectorOp::Delete(vector_id) => {
                            // For now, we'll just log the deletion
                            // The actual removal from indexes will be handled during commit
                            log::info!("Processing delete operation for vector_id: {}", vector_id);
                            Ok::<_, WaCustomError>(())
                        }
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
        txn.pre_commit(collection, config)?;
        update_background_version(&collection.lmdb, version)?;
        fs::remove_file(collection.get_path().join(format!("{}.wal", *version)))
            .map_err(BufIoError::Io)
            .unwrap();
        collection.vcs.update_version_metadata(
            version,
            wal.records_upserted(),
            wal.records_deleted(),
            wal.total_operations(),
        )?;
        collection.is_indexing.store(false, Ordering::Relaxed);
        Ok(())
    }

    pub fn index_version_on_restart(
        collection: &Collection,
        config: &Config,
        threadpool: &ThreadPool,
        version: VersionNumber,
    ) -> Result<(), WaCustomError> {
        let version_info = collection.vcs.get_version(version)?;
        match version_info.source {
            VersionSource::Explicit { transaction_id } => {
                Self::index_explicit_txn(collection, config, threadpool, transaction_id, version)?;
            }
            VersionSource::Implicit { .. } => {
                Self::index_implicit_txn(collection, config, threadpool, version)?;
            }
        }

        Ok(())
    }

    pub fn implicit_txn_upsert(
        collection: &Collection,
        transaction: &ImplicitTransaction,
        config: &Config,
        embeddings: Vec<RawVectorEmbedding>,
    ) -> Result<(), WaCustomError> {
        let version = transaction.version(collection)?;
        transaction.append_to_wal(collection, VectorOp::Upsert(embeddings.clone()))?;
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
        use std::collections::HashSet;
        let mut seen_ids = HashSet::new();
        for (_hash, arc_quotient) in collection.external_to_internal_map.root.quotients.map.to_list() {
            let internal_id = arc_quotient.value.read().value;
            seen_ids.insert(internal_id);
        }
        let vector_count = seen_ids.len() as u32;
        collection.vcs.update_version_metadata(
            version,
            vector_count,
            0,
            vector_count,
        ).map_err(|e| WaCustomError::DatabaseError(format!("Failed to update version metadata: {e}")))?;
        Ok(())
    }

    pub fn implicit_txn_delete(
        collection: &Collection,
        transaction: &ImplicitTransaction,
        _config: &Config,
        vector_id: VectorId,
    ) -> Result<(), WaCustomError> {
        let version = transaction.version(collection)?;
        transaction.append_to_wal(collection, VectorOp::Delete(vector_id.clone()))?;
        
        // For now, we'll just log the deletion
        // The actual removal from indexes will be handled during commit
        log::info!("Processing implicit transaction delete for vector_id: {}", vector_id);
        
        // Update version metadata to reflect the deletion
        use std::collections::HashSet;
        let mut seen_ids = HashSet::new();
        for (_hash, arc_quotient) in collection.external_to_internal_map.root.quotients.map.to_list() {
            let internal_id = arc_quotient.value.read().value;
            seen_ids.insert(internal_id);
        }
        let vector_count = seen_ids.len() as u32;
        collection.vcs.update_version_metadata(
            version,
            vector_count,
            1, // records_deleted
            vector_count + 1, // total_operations (including the delete)
        ).map_err(|e| WaCustomError::DatabaseError(format!("Failed to update version metadata: {e}")))?;
        
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
