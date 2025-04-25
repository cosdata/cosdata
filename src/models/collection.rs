use super::buffered_io::BufferManagerFactory;
use super::collection_transaction::CollectionTransaction;
use super::common::WaCustomError;
use super::meta_persist::store_highest_internal_id;
use super::paths::get_data_path;
use super::tree_map::{TreeMap, TreeMapVec};
use super::types::{get_collections_path, DocumentId, InternalId, MetaDb, VectorId};
use super::versioning::{Hash, VersionControl};
use crate::config_loader::Config;
use crate::indexes::hnsw::{DenseInputEmbedding, HNSWIndex};
use crate::indexes::inverted::types::SparsePair;
use crate::indexes::inverted::{InvertedIndex, SparseInputEmbedding};
use crate::indexes::tf_idf::{TFIDFIndex, TFIDFInputEmbedding};
use crate::indexes::IndexOps;
use crate::metadata::{MetadataFields, MetadataSchema};
use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use serde_cbor::to_vec;
use siphasher::sip::SipHasher24;
use std::fs::create_dir_all;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;
use std::{fs, hash::Hasher, path::Path, sync::Arc};

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct DenseVectorOptions {
    pub enabled: bool,
    pub dimension: usize,
}

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct SparseVectorOptions {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFIDFOptions {
    pub enabled: bool,
}

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct CollectionConfig {
    pub max_vectors: Option<u32>,
    pub replication_factor: Option<u32>,
}

#[derive(Clone)]
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

pub struct Collection {
    pub meta: CollectionMetadata,
    pub lmdb: MetaDb,
    pub current_version: RwLock<Hash>,
    pub current_open_transaction: RwLock<Option<CollectionTransaction>>,
    pub vcs: VersionControl,
    pub internal_to_external_map: TreeMap<InternalId, RawVectorEmbedding>,
    pub external_to_internal_map: TreeMap<VectorId, InternalId>,
    pub document_to_internals_map: TreeMapVec<DocumentId, InternalId>,
    pub internal_id_counter: AtomicU32,
    pub hnsw_index: RwLock<Option<Arc<HNSWIndex>>>,
    pub inverted_index: RwLock<Option<Arc<InvertedIndex>>>,
    pub tf_idf_index: RwLock<Option<Arc<TFIDFIndex>>>,
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
        config: CollectionConfig,
        store_raw_text: bool,
        lmdb: MetaDb,
        current_version: Hash,
        vcs: VersionControl,
    ) -> Result<Self, WaCustomError> {
        if name.is_empty() {
            return Err(WaCustomError::InvalidParams);
        }

        let collections_path = get_collections_path().join(&name);

        let internal_to_external_map_bufmans = BufferManagerFactory::new(
            collections_path.clone().into(),
            |root, part| root.join(format!("{}.itoe", part)),
            8192,
        );

        let external_to_internal_map_bufmans = BufferManagerFactory::new(
            collections_path.clone().into(),
            |root, part| root.join(format!("{}.etoi", part)),
            8192,
        );

        let document_to_internals_map_bufmans = BufferManagerFactory::new(
            collections_path.into(),
            |root, part| root.join(format!("{}.dtoi", part)),
            8192,
        );

        let collection = Collection {
            meta: CollectionMetadata {
                name,
                description,
                dense_vector: dense_vector_options,
                sparse_vector: sparse_vector_options,
                tf_idf_options,
                metadata_schema,
                config,
                store_raw_text,
            },
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: RwLock::new(None),
            vcs,
            internal_to_external_map: TreeMap::new(internal_to_external_map_bufmans),
            external_to_internal_map: TreeMap::new(external_to_internal_map_bufmans),
            document_to_internals_map: TreeMapVec::new(document_to_internals_map_bufmans),
            internal_id_counter: AtomicU32::new(0),
            hnsw_index: RwLock::new(None),
            inverted_index: RwLock::new(None),
            tf_idf_index: RwLock::new(None),
        };

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
        self.hnsw_index.read().unwrap().clone()
    }

    pub fn get_inverted_index(&self) -> Option<Arc<InvertedIndex>> {
        self.inverted_index.read().unwrap().clone()
    }

    pub fn get_tf_idf_index(&self) -> Option<Arc<TFIDFIndex>> {
        self.tf_idf_index.read().unwrap().clone()
    }

    pub fn run_upload(
        &self,
        embeddings: Vec<RawVectorEmbedding>,
        transaction: &CollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        let id_start = self
            .internal_id_counter
            .fetch_add(embeddings.len() as u32, Ordering::Relaxed);

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

                    let internal_id = InternalId::from(id_start + i as u32);

                    if let Some(values) = dense_values {
                        acc.0
                            .push(DenseInputEmbedding(internal_id, values, metadata, false));
                    } else if let Some(values) = sparse_values {
                        acc.1.push(SparseInputEmbedding(internal_id, values));
                    } else if let Some(text) = text {
                        acc.2.push(TFIDFInputEmbedding(internal_id, text));
                    }

                    if !self.meta.store_raw_text {
                        embedding.text = None;
                    }

                    self.internal_to_external_map
                        .insert(transaction.id, internal_id, embedding);
                    self.external_to_internal_map
                        .insert(transaction.id, id, internal_id);

                    if let Some(document_id) = document_id {
                        self.document_to_internals_map.push(
                            transaction.id,
                            document_id,
                            internal_id,
                        );
                    }

                    acc
                },
            );

        if !dense_embs.is_empty() {
            if let Some(hnsw_index) = &*self.hnsw_index.read().unwrap() {
                hnsw_index.run_upload(self, dense_embs, transaction, config)?;
            }
        }

        if !sparse_embs.is_empty() {
            if let Some(inverted_index) = &*self.inverted_index.read().unwrap() {
                inverted_index.run_upload(self, sparse_embs, transaction, config)?;
            }
        }

        if !tf_idf_embs.is_empty() {
            if let Some(tf_idf_index) = &*self.tf_idf_index.read().unwrap() {
                tf_idf_index.run_upload(self, tf_idf_embs, transaction, config)?;
            }
        }

        Ok(())
    }

    pub fn flush(&self, config: &Config) -> Result<(), WaCustomError> {
        self.internal_to_external_map
            .serialize(config.tree_map_serialized_parts)?;
        self.external_to_internal_map
            .serialize(config.tree_map_serialized_parts)?;
        self.document_to_internals_map
            .serialize(config.tree_map_serialized_parts)?;
        store_highest_internal_id(&self.lmdb, self.internal_id_counter.load(Ordering::Relaxed))?;
        Ok(())
    }
}
