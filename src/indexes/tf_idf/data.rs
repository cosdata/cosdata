use std::hash::Hasher;

use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde_cbor::{from_slice, to_vec};
use siphasher::sip::SipHasher24;

use crate::models::common::WaCustomError;

use super::TFIDFIndex;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TFIDFIndexData {
    pub name: String,
    pub description: Option<String>,
    pub max_vectors: Option<i32>,
    pub sample_threshold: usize,
    pub store_raw_text: bool,
    pub k1: f32,
    pub b: f32,
}

impl From<&TFIDFIndex> for TFIDFIndexData {
    fn from(tf_idf_index: &TFIDFIndex) -> Self {
        Self {
            name: tf_idf_index.name.clone(),
            description: tf_idf_index.description.clone(),
            max_vectors: tf_idf_index.max_vectors,
            sample_threshold: tf_idf_index.sample_threshold,
            store_raw_text: tf_idf_index.store_raw_text,
            k1: tf_idf_index.k1,
            b: tf_idf_index.b,
        }
    }
}

impl TFIDFIndexData {
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

    /// loads TF-IDF index data for a collection
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

    /// persists TF-IDF index data for a collection
    pub fn persist(
        env: &Environment,
        db: Database,
        tf_idf_index: &TFIDFIndex,
    ) -> Result<(), WaCustomError> {
        let data = Self::from(tf_idf_index);

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

    /// deletes TF-IDF index for a collection
    pub fn delete_index(
        env: &Environment,
        db: Database,
        tf_idf_index: &TFIDFIndex,
    ) -> lmdb::Result<()> {
        let key = Self::get_key_for_name(&tf_idf_index.name);
        let mut txn = env.begin_rw_txn()?;
        txn.del(db, &key, None)?;
        txn.commit()?;
        Ok(())
    }
}
