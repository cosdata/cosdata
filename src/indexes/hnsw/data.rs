use std::hash::Hasher;

use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde_cbor::{from_slice, to_vec};
use siphasher::sip::SipHasher24;

use crate::{
    models::{
        common::WaCustomError,
        prob_lazy_load::lazy_item::FileIndex,
        types::{DistanceMetric, QuantizationMetric},
    },
    quantization::StorageType,
};

use super::{types::HNSWHyperParams, HNSWIndex};

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct HNSWIndexData {
    pub name: String,
    pub hnsw_params: HNSWHyperParams,
    pub levels_prob: Vec<(f64, u8)>,
    pub dim: usize,
    pub file_index: FileIndex,
    pub quantization_metric: QuantizationMetric,
    pub distance_metric: DistanceMetric,
    pub storage_type: StorageType,
    pub size: usize,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
    pub sample_threshold: usize,
}

impl TryFrom<&HNSWIndex> for HNSWIndexData {
    type Error = WaCustomError;

    fn try_from(hnsw_index: &HNSWIndex) -> Result<Self, Self::Error> {
        let offset = hnsw_index.root_vec_offset();
        let hnsw_index_data = Self {
            name: hnsw_index.name.clone(),
            hnsw_params: hnsw_index.hnsw_params.read().unwrap().clone(),
            levels_prob: hnsw_index.levels_prob.clone(),
            dim: hnsw_index.dim,
            file_index: offset,
            quantization_metric: hnsw_index.quantization_metric.read().unwrap().clone(),
            distance_metric: *hnsw_index.distance_metric.read().unwrap(),
            storage_type: *hnsw_index.storage_type.read().unwrap(),
            size: 0,
            lower_bound: None,
            upper_bound: None,
            sample_threshold: hnsw_index.sample_threshold,
        };
        Ok(hnsw_index_data)
    }
}

impl HNSWIndexData {
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

    /// loads HNSW index data for a collection
    pub fn load(env: &Environment, db: Database, collection_id: &[u8; 8]) -> lmdb::Result<Self> {
        let txn = env.begin_ro_txn().unwrap();
        let index = txn.get(db, collection_id)?;
        let index = from_slice(index).unwrap();
        Ok(index)
    }

    /// persists HNSW index data for a collection
    pub fn persist(
        env: &Environment,
        db: Database,
        hnsw_index: &HNSWIndex,
    ) -> Result<(), WaCustomError> {
        let data = Self::try_from(hnsw_index)?;

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

    /// deletes HNSW index for a collection
    pub fn delete_index(
        env: &Environment,
        db: Database,
        hnsw_index: &HNSWIndex,
    ) -> lmdb::Result<()> {
        let key = Self::get_key_for_name(&hnsw_index.name);
        let mut txn = env.begin_rw_txn()?;
        txn.del(db, &key, None)?;
        txn.commit()?;
        Ok(())
    }
}
