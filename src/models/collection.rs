use super::buffered_io::BufferManagerFactory;
use super::collection_transaction::{
    BackgroundCollectionTransaction, CollectionTransaction, TransactionStatus,
};
use super::common::WaCustomError;
use super::indexing_manager::IndexingManager;
use super::meta_persist::store_highest_internal_id;
use super::paths::get_data_path;
use super::tree_map::{TreeMap, TreeMapVec};
use super::types::{get_collections_path, DocumentId, InternalId, MetaDb, VectorId};
use super::versioning::{VersionControl, VersionNumber};
use super::wal::VectorOp;
use crate::app_context::AppContext;
use crate::config_loader::Config;
use crate::indexes::hnsw::{DenseInputEmbedding, HNSWIndex};
use crate::indexes::inverted::types::SparsePair;
use crate::indexes::inverted::{InvertedIndex, SparseInputEmbedding};
use crate::indexes::tf_idf::{TFIDFIndex, TFIDFInputEmbedding};
use crate::indexes::IndexOps;
use crate::metadata::{MetadataFields, MetadataSchema};
use chrono::{DateTime, TimeZone, Utc};
use lmdb::{Database, Environment, Transaction, WriteFlags};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_cbor::to_vec;
use siphasher::sip::SipHasher24;
use std::fs::create_dir_all;
use std::sync::atomic::{AtomicU32, Ordering};
use std::{fs, hash::Hasher, path::Path, sync::Arc};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DenseVectorOptions {
    pub enabled: bool,
    pub dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SparseVectorOptions {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TFIDFOptions {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CollectionConfig {
    pub max_vectors: Option<u32>,
    pub replication_factor: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct RawVectorEmbedding {
    pub id: VectorId,
    pub document_id: Option<DocumentId>,
    pub dense_values: Option<Vec<f32>>,
    pub metadata: Option<MetadataFields>,
    pub sparse_values: Option<Vec<SparsePair>>,
    pub text: Option<String>,
}

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct CollectionMetadata {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub tf_idf_options: TFIDFOptions,
    pub metadata_schema: Option<MetadataSchema>,
    pub config: CollectionConfig,
    pub store_raw_text: bool,
}

#[derive(Debug, Serialize)]
pub struct CollectionIndexingStatusSummary {
    pub total_transactions: u32,
    pub completed_transactions: u32,
    pub in_progress_transactions: u32,
    pub not_started_transactions: u32,
    pub total_records_indexed_completed: u64,
    pub average_rate_per_second_completed: f32,
}

#[derive(Debug, Serialize)]
pub struct TransactionStatusWithTransactionId {
    pub transaction_id: VersionNumber,
    #[serde(flatten)]
    pub status: TransactionStatus,
}

#[derive(Debug, Serialize)]
pub struct CollectionIndexingStatus {
    pub collection_name: String,
    pub status_summary: CollectionIndexingStatusSummary,
    pub active_transactions: Vec<TransactionStatusWithTransactionId>,
    pub last_synced: DateTime<Utc>,
}

pub struct Collection {
    pub meta: CollectionMetadata,
    pub lmdb: MetaDb,
    pub current_version: RwLock<VersionNumber>,
    pub current_open_transaction: RwLock<Option<CollectionTransaction>>,
    pub vcs: VersionControl,
    pub internal_to_external_map: TreeMap<InternalId, RawVectorEmbedding>,
    pub external_to_internal_map: TreeMap<VectorId, InternalId>,
    pub document_to_internals_map: TreeMapVec<DocumentId, InternalId>,
    pub transaction_status_map: TreeMap<VersionNumber, RwLock<TransactionStatus>>,
    pub internal_id_counter: AtomicU32,
    pub hnsw_index: RwLock<Option<Arc<HNSWIndex>>>,
    pub inverted_index: RwLock<Option<Arc<InvertedIndex>>>,
    pub tf_idf_index: RwLock<Option<Arc<TFIDFIndex>>>,
    // this field is actually NOT optional, the only reason it is wrapped in
    // `Option` is to allow us to create `Collection` first without the
    // indexing manager, because `IndexingManager`'s constructor also requires
    // a reference to the collection
    pub indexing_manager: RwLock<Option<IndexingManager>>,
}

impl Collection {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        description: Option<String>,
        dense_vector_options: DenseVectorOptions,
        sparse_vector_options: SparseVectorOptions,
        tf_idf_options: TFIDFOptions,
        metadata_schema: Option<MetadataSchema>,
        collection_config: CollectionConfig,
        store_raw_text: bool,
        lmdb: MetaDb,
        current_version: VersionNumber,
        vcs: VersionControl,
        ctx: &AppContext,
    ) -> Result<Arc<Self>, WaCustomError> {
        if name.is_empty() {
            return Err(WaCustomError::InvalidParams);
        }

        let collections_path: Arc<Path> = get_collections_path().join(&name).into();

        let internal_to_external_map_bufmans = BufferManagerFactory::new(
            collections_path.clone(),
            |root, part| root.join(format!("{}.itoe", part)),
            8192,
        );

        let external_to_internal_map_bufmans = BufferManagerFactory::new(
            collections_path.clone(),
            |root, part| root.join(format!("{}.etoi", part)),
            8192,
        );

        let document_to_internals_map_bufmans = BufferManagerFactory::new(
            collections_path.clone(),
            |root, part| root.join(format!("{}.dtoi", part)),
            8192,
        );

        let transaction_status_map_bufmans = BufferManagerFactory::new(
            collections_path,
            |root, part| root.join(format!("{}.txn_status", part)),
            8192,
        );

        let collection = Arc::new(Collection {
            meta: CollectionMetadata {
                name,
                description,
                dense_vector: dense_vector_options,
                sparse_vector: sparse_vector_options,
                tf_idf_options,
                metadata_schema,
                config: collection_config,
                store_raw_text,
            },
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: RwLock::new(None),
            vcs,
            internal_to_external_map: TreeMap::new(internal_to_external_map_bufmans),
            external_to_internal_map: TreeMap::new(external_to_internal_map_bufmans),
            document_to_internals_map: TreeMapVec::new(document_to_internals_map_bufmans),
            transaction_status_map: TreeMap::new(transaction_status_map_bufmans),
            internal_id_counter: AtomicU32::new(0),
            hnsw_index: RwLock::new(None),
            inverted_index: RwLock::new(None),
            tf_idf_index: RwLock::new(None),
            indexing_manager: RwLock::new(None),
        });

        *collection.indexing_manager.write() = Some(IndexingManager::new(
            collection.clone(),
            ctx.config.clone(),
            ctx.threadpool.clone(),
        ));

        let collection_path = collection.get_path();
        fs::create_dir_all(&collection_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

        Ok(collection)
    }

    /// Computes the SipHash of the collection name
    pub fn get_hash(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        hasher.write(self.meta.name.as_bytes());
        hasher.finish()
    }

    /// computes the key used to store the collection in the database
    pub fn get_key(&self) -> [u8; 8] {
        let hash = self.get_hash();
        hash.to_le_bytes()
    }

    /// creates a path out of the collection name
    pub fn get_path(&self) -> Arc<Path> {
        let collections_path = get_data_path().join("collections");
        create_dir_all(&collections_path).expect("Failed to create collections directory");
        collections_path.join(&self.meta.name).into()
    }

    /// serializes the collection
    pub fn serialize(&self) -> Result<Vec<u8>, WaCustomError> {
        to_vec(&self.meta).map_err(|e| WaCustomError::SerializationError(e.to_string()))
    }

    /// perists the collection instance on disk (lmdb -> collections database)
    pub fn persist(&self, env: &Environment, db: Database) -> Result<(), WaCustomError> {
        let key = self.get_key();
        let value = self.serialize()?;

        let mut txn = env
            .begin_rw_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.put(db, &key, &value, WriteFlags::empty())
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
        txn.commit()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// deletes a collection instance from the disk (lmdb -> collections database)
    #[allow(dead_code)]
    pub fn delete(&self, env: &Environment, db: Database) -> Result<(), WaCustomError> {
        let key = self.get_key();

        let mut txn = env
            .begin_rw_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.del(db, &key, None)
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
        txn.commit()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    pub fn get_hnsw_index(&self) -> Option<Arc<HNSWIndex>> {
        self.hnsw_index.read().clone()
    }

    pub fn get_inverted_index(&self) -> Option<Arc<InvertedIndex>> {
        self.inverted_index.read().clone()
    }

    pub fn get_tf_idf_index(&self) -> Option<Arc<TFIDFIndex>> {
        self.tf_idf_index.read().clone()
    }

    /// Returns the raw embedding mapped to an internal id
    ///
    /// It's recommended to call this method instead of directly
    /// accessing the 'internal_to_external' field of this
    /// struct. This method specially handles the case of the
    /// collection supporting metadata filtering on hnsw index, in
    /// which case not all internal_ids may be mapped with an
    /// embedding. There are mappings only for base node ids. For
    /// metadata replica nodes, this method first finds the base_node
    /// id and uses that to obtain the raw vector embedding
    pub fn get_raw_emb_by_internal_id(
        &self,
        internal_id: &InternalId,
    ) -> Option<&RawVectorEmbedding> {
        let mapped_internal_id = if let Some(hnsw_index) = self.get_hnsw_index() {
            if self.meta.metadata_schema.is_some() {
                let id = **internal_id;
                InternalId::from(id - id % hnsw_index.max_replica_per_node as u32)
            } else {
                *internal_id
            }
        } else {
            *internal_id
        };
        self.internal_to_external_map
            .get_latest(&mapped_internal_id)
    }

    pub fn run_upload(
        &self,
        embeddings: Vec<RawVectorEmbedding>,
        transaction: &CollectionTransaction,
    ) -> Result<(), WaCustomError> {
     
       // Check if any of the IDs already exist in the transaction
        for embedding in &embeddings {
            if self.external_to_internal_map.get_latest(&embedding.id).is_some() {
                return Err(WaCustomError::InvalidData(format!(
                    "Vector ID already exists: {}",
                    embedding.id
                )));
            }
        }

        for embedding in embeddings.clone() {
            if let Some(dense_values) = embedding.dense_values {
                if let Some(hnsw_index) = self.get_hnsw_index() {
                    let dense_emb = DenseInputEmbedding(
                        InternalId::from(u32::MAX),
                        dense_values,
                        embedding.metadata,
                        false,
                    );
                    hnsw_index.validate_embedding(dense_emb)?;
                }
            }

            if let Some(sparse_values) = embedding.sparse_values {
                if let Some(inverted_index) = self.get_inverted_index() {
                    let sparse_emb =
                        SparseInputEmbedding(InternalId::from(u32::MAX), sparse_values);
                    inverted_index.validate_embedding(sparse_emb)?;
                }
            }

            if let Some(text) = embedding.text {
                if let Some(tf_idf_index) = self.get_tf_idf_index() {
                    let tf_idf_emb = TFIDFInputEmbedding(InternalId::from(u32::MAX), text);
                    tf_idf_index.validate_embedding(tf_idf_emb)?;
                }
            }
        }

        transaction.wal.append(VectorOp::Upsert(embeddings))?;

        Ok(())
    }

    pub fn index_embeddings(
        &self,
        embeddings: Vec<RawVectorEmbedding>,
        transaction: &BackgroundCollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        let num_nodes_per_emb = if let Some(hnsw_index) = &*self.hnsw_index.read() {
            hnsw_index.max_replica_per_node as usize
        } else {
            1
        };
        let num_ids_to_reserve = embeddings.len() * num_nodes_per_emb;
        let id_start = self
            .internal_id_counter
            .fetch_add(num_ids_to_reserve as u32, Ordering::Relaxed);

        let (dense_embs, sparse_embs, tf_idf_embs): (Vec<_>, Vec<_>, Vec<_>) =
            embeddings.into_iter().enumerate().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |mut acc, (i, mut embedding)| {
                    let RawVectorEmbedding {
                        id,
                        document_id,
                        dense_values,
                        metadata,
                        sparse_values,
                        text,
                    } = embedding.clone();

                    let internal_id = InternalId::from(id_start + (i * num_nodes_per_emb) as u32);

                    if let Some(values) = dense_values {
                        acc.0
                            .push(DenseInputEmbedding(internal_id, values, metadata, false));
                    }
                    if let Some(values) = sparse_values {
                        acc.1.push(SparseInputEmbedding(internal_id, values));
                    }
                    if let Some(text) = text {
                        acc.2.push(TFIDFInputEmbedding(internal_id, text));
                    }

                    if !self.meta.store_raw_text {
                        embedding.text = None;
                    }

                    self.internal_to_external_map.insert(
                        transaction.version,
                        &internal_id,
                        embedding,
                    );
                    self.external_to_internal_map
                        .insert(transaction.version, &id, internal_id);

                    if let Some(document_id) = document_id {
                        self.document_to_internals_map.push(
                            transaction.version,
                            &document_id,
                            internal_id,
                        );
                    }

                    acc
                },
            );

        if !dense_embs.is_empty() {
            if let Some(hnsw_index) = &*self.hnsw_index.read() {
                hnsw_index.run_upload(self, dense_embs, transaction, config)?;
            }
        }

        if !sparse_embs.is_empty() {
            if let Some(inverted_index) = &*self.inverted_index.read() {
                inverted_index.run_upload(self, sparse_embs, transaction, config)?;
            }
        }

        if !tf_idf_embs.is_empty() {
            if let Some(tf_idf_index) = &*self.tf_idf_index.read() {
                tf_idf_index.run_upload(self, tf_idf_embs, transaction, config)?;
            }
        }

        Ok(())
    }

    pub fn trigger_indexing(&self, version: VersionNumber) {
        self.transaction_status_map.insert(
            version,
            &version,
            RwLock::new(TransactionStatus::NotStarted {
                last_updated: Utc::now(),
            }),
        );
        self.indexing_manager
            .read()
            .as_ref()
            .unwrap()
            .trigger(version);
    }

    pub fn get_latest_transaction_status(&self) -> Option<TransactionStatus> {
        Some(
            self.transaction_status_map
                .get_latest(&self.current_version.read())?
                .read()
                .clone(),
        )
    }

    pub fn flush(&self, config: &Config) -> Result<(), WaCustomError> {
        self.internal_to_external_map
            .serialize(config.tree_map_serialized_parts)?;
        self.external_to_internal_map
            .serialize(config.tree_map_serialized_parts)?;
        self.document_to_internals_map
            .serialize(config.tree_map_serialized_parts)?;
        self.transaction_status_map
            .serialize(config.tree_map_serialized_parts)?;
        store_highest_internal_id(&self.lmdb, self.internal_id_counter.load(Ordering::Relaxed))?;
        Ok(())
    }

    pub fn indexing_status(&self) -> Result<CollectionIndexingStatus, WaCustomError> {
        let mut active_transactions = Vec::new();
        let mut last_synced = Utc.timestamp_opt(0, 0).unwrap();
        let mut total_transactions = 0;
        let mut completed_transactions = 0;
        let mut in_progress_transactions = 0;
        let mut not_started_transactions = 0;
        let mut total_records_indexed_completed = 0u64;
        let mut rate_per_second_acc = 0.0;

        for version_info in self.vcs.get_versions()? {
            let Some(status) = self
                .transaction_status_map
                .get_latest(&version_info.version)
            else {
                continue;
            };
            total_transactions += 1;
            let status = status.read();
            match &*status {
                TransactionStatus::NotStarted { last_updated } => {
                    last_synced = last_synced.max(*last_updated);
                    not_started_transactions += 1;
                }
                TransactionStatus::InProgress {
                    progress,
                    last_updated,
                } => {
                    last_synced = last_synced.max(*last_updated);
                    total_records_indexed_completed += progress.records_indexed as u64;
                    rate_per_second_acc += progress.rate_per_second;
                    in_progress_transactions += 1;
                }
                TransactionStatus::Complete {
                    summary,
                    last_updated,
                } => {
                    last_synced = last_synced.max(*last_updated);
                    total_records_indexed_completed += summary.total_records_indexed as u64;
                    rate_per_second_acc += summary.average_rate_per_second;
                    completed_transactions += 1;
                }
            }

            if !matches!(&*status, TransactionStatus::Complete { .. }) {
                active_transactions.push(TransactionStatusWithTransactionId {
                    transaction_id: version_info.version,
                    status: status.clone(),
                });
            }
        }

        Ok(CollectionIndexingStatus {
            collection_name: self.meta.name.clone(),
            status_summary: CollectionIndexingStatusSummary {
                total_transactions,
                completed_transactions,
                in_progress_transactions,
                not_started_transactions,
                total_records_indexed_completed,
                average_rate_per_second_completed: rate_per_second_acc
                    / (in_progress_transactions + completed_transactions) as f32,
            },
            active_transactions,
            last_synced,
        })
    }
}
