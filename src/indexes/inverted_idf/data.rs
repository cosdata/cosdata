use std::hash::Hasher;

use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde_cbor::{from_slice, to_vec};
use siphasher::sip::SipHasher24;

use crate::models::common::WaCustomError;

use super::InvertedIndexIDF;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct InvertedIndexIDFData {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub max_vectors: Option<i32>,
}

impl From<&InvertedIndexIDF> for InvertedIndexIDFData {
    fn from(inverted_index_idf: &InvertedIndexIDF) -> Self {
        Self {
            name: inverted_index_idf.name.clone(),
            description: inverted_index_idf.description.clone(),
            auto_create_index: inverted_index_idf.auto_create_index,
            max_vectors: inverted_index_idf.max_vectors,
        }
    }
}

impl InvertedIndexIDFData {
    /// Computes the SipHash of a collection/index name
    pub fn get_hash_for_name(name: &str) -> u64 {
        // Compute SipHash of the collection name
        let mut hasher = SipHasher24::new();
        hasher.write(name.as_bytes());

        hasher.finish()
    }

    /// computes the key used to store an index in the database
    pub fn get_key_for_name(name: &str) -> [u8; 8] {
        let hash = Self::get_hash_for_name(name);

        hash.to_le_bytes()
    }

    /// computes the key used to store the collection in the database
    pub fn get_key(&self) -> [u8; 8] {
        Self::get_key_for_name(&self.name)
    }

    /// loads inverted index data for a collection
    pub fn load(
        env: &Environment,
        db: Database,
        collection_id: &[u8; 8],
    ) -> lmdb::Result<Option<Self>> {
        let txn = env.begin_ro_txn().unwrap();
        let index = match txn.get(db, collection_id) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        };
        let index: Self = from_slice(index).unwrap();
        Ok(Some(index))
    }

    /// persists inverted index data for a collection
    pub fn persist(
        env: &Environment,
        db: Database,
        inverted_index: &InvertedIndexIDF,
    ) -> Result<(), WaCustomError> {
        let data = Self::from(inverted_index);

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
        inverted_index: &InvertedIndexIDF,
    ) -> lmdb::Result<()> {
        let key = Self::get_key_for_name(&inverted_index.name);
        let mut txn = env.begin_rw_txn()?;
        txn.del(db, &key, None)?;
        txn.commit()?;
        Ok(())
    }
}
