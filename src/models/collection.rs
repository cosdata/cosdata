use super::collection_transaction::CollectionTransaction;
use super::common::WaCustomError;
use super::paths::get_data_path;
use super::types::MetaDb;
use super::versioning::{Hash, VersionControl};
use crate::indexes::hnsw::HNSWIndex;
use crate::indexes::inverted::InvertedIndex;
use crate::indexes::tf_idf::TFIDFIndex;
use crate::metadata::MetadataSchema;
use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use serde_cbor::to_vec;
use siphasher::sip::SipHasher24;
use std::fs::create_dir_all;
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

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct CollectionMetadata {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub tf_idf_options: TFIDFOptions,
    pub metadata_schema: Option<MetadataSchema>,
    pub config: CollectionConfig,
}

pub struct Collection {
    pub meta: CollectionMetadata,
    pub lmdb: MetaDb,
    pub current_version: RwLock<Hash>,
    pub current_open_transaction: RwLock<Option<CollectionTransaction>>,
    pub vcs: VersionControl,
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
        lmdb: MetaDb,
        current_version: Hash,
        vcs: VersionControl,
    ) -> Result<Self, WaCustomError> {
        if name.is_empty() {
            return Err(WaCustomError::InvalidParams);
        }

        let collection = Collection {
            meta: CollectionMetadata {
                name,
                description,
                dense_vector: dense_vector_options,
                sparse_vector: sparse_vector_options,
                tf_idf_options,
                metadata_schema,
                config,
            },
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: RwLock::new(None),
            vcs,
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
}
