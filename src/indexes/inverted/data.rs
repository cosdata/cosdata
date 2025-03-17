use std::hash::Hasher;

use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde_cbor::{from_slice, to_vec};
use siphasher::sip::SipHasher24;

use crate::models::common::WaCustomError;

use super::InvertedIndex;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct InvertedIndexData {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub metadata_schema: Option<String>,
    pub max_vectors: Option<i32>,
    pub quantization_bits: u8,
    pub sample_threshold: usize,
    pub early_terminate_threshold: f32,
}

impl TryFrom<&InvertedIndex> for InvertedIndexData {
    type Error = WaCustomError;

    fn try_from(inverted_index: &InvertedIndex) -> Result<Self, Self::Error> {
        let inverted_index_data = Self {
            name: inverted_index.name.clone(),
            description: inverted_index.description.clone(),
            auto_create_index: inverted_index.auto_create_index,
            metadata_schema: inverted_index.metadata_schema.clone(),
            max_vectors: inverted_index.max_vectors,
            quantization_bits: inverted_index.root.root.quantization_bits,
            sample_threshold: inverted_index.sample_threshold,
            early_terminate_threshold: inverted_index.early_terminate_threshold,
        };
        Ok(inverted_index_data)
    }
}

impl InvertedIndexData {
    /// Computes the SipHash of a collection/index name
    pub fn get_hash_for_name(name: &str) -> u64 {
        // Compute SipHash of the collection name
        let mut hasher = SipHasher24::new();
        hasher.write(name.as_bytes());

        hasher.finish()
    }

    /// computes the key used to store an index in the database
    pub fn get_key_for_name(name: &str) -> [u8; 8] {
        let hash = InvertedIndexData::get_hash_for_name(name);

        hash.to_le_bytes()
    }

    /// computes the key used to store the collection in the database
    pub fn get_key(&self) -> [u8; 8] {
        InvertedIndexData::get_key_for_name(&self.name)
    }

    /// loads inverted index data for a collection
    pub fn load(
        env: &Environment,
        db: Database,
        collection_id: &[u8; 8],
    ) -> lmdb::Result<InvertedIndexData> {
        let txn = env.begin_ro_txn().unwrap();
        let index = txn.get(db, collection_id)?;
        let index: InvertedIndexData = from_slice(index).unwrap();
        Ok(index)
    }

    /// persists inverted index data for a collection
    pub fn persist(
        env: &Environment,
        db: Database,
        inverted_index: &InvertedIndex,
    ) -> Result<(), WaCustomError> {
        let data = InvertedIndexData::try_from(inverted_index)?;

        // Compute SipHash of the collection name
        let key = data.get_key();

        let val = to_vec(&data).map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

        let mut txn = env
            .begin_rw_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.put(db, &key, &val, WriteFlags::empty())
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.commit()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// deletes inverted index for a collection
    pub fn delete_index(
        env: &Environment,
        db: Database,
        inverted_index: &InvertedIndex,
    ) -> lmdb::Result<()> {
        let key = InvertedIndexData::get_key_for_name(&inverted_index.name);
        let mut txn = env.begin_rw_txn()?;
        txn.del(db, &key, None)?;
        txn.commit()?;
        Ok(())
    }
}
