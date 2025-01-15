use crate::models::common::*;
use crate::models::types::*;
use crate::models::versioning::*;
use crate::quantization::StorageType;
use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use serde_cbor::{from_slice, to_vec};
use siphasher::sip::SipHasher24;
use std::hash::Hasher;
use std::sync::Arc;

use super::collection::Collection;
use super::lazy_load::FileIndex;

/// updates the current version of a collection
pub fn update_current_version(lmdb: &MetaDb, version_hash: Hash) -> Result<(), WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let bytes = version_hash.to_le_bytes();

    txn.put(*db, &"current_version", &bytes, WriteFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(())
}

/// retrieves the current version of a collection
pub fn retrieve_current_version(lmdb: &MetaDb) -> Result<Hash, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let serialized_hash = txn.get(*db, &"current_version").map_err(|e| match e {
        lmdb::Error::NotFound => {
            WaCustomError::DatabaseError("Record not found: current_version".to_string())
        }
        _ => WaCustomError::DatabaseError(e.to_string()),
    })?;

    let bytes: [u8; 4] = serialized_hash.try_into().map_err(|_| {
        WaCustomError::DeserializationError(
            "Failed to deserialize Hash: length mismatch".to_string(),
        )
    })?;
    let hash = Hash::from(u32::from_le_bytes(bytes));

    Ok(hash)
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DenseIndexData {
    pub name: String,
    pub hnsw_params: HNSWHyperParams,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub dim: usize,
    pub file_index: FileIndex,
    pub quantization_metric: QuantizationMetric,
    pub distance_metric: DistanceMetric,
    pub storage_type: StorageType,
    pub size: usize,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
}

impl TryFrom<Arc<DenseIndex>> for DenseIndexData {
    type Error = WaCustomError;
    fn try_from(dense_index: Arc<DenseIndex>) -> Result<Self, Self::Error> {
        let offset = dense_index
            .root_vec_offset()
            .ok_or(WaCustomError::NodeError(
                "FileIndex must be set for root node".to_owned(),
            ))?;
        if let FileIndex::Invalid = offset {
            return Err(WaCustomError::NodeError(
                "FileIndex must be valid for root node".to_owned(),
            ));
        };
        let dense_index_data = Self {
            name: dense_index.database_name.clone(),
            hnsw_params: dense_index.hnsw_params.read().unwrap().clone(),
            levels_prob: dense_index.levels_prob.clone(),
            dim: dense_index.dim,
            file_index: offset,
            quantization_metric: dense_index.quantization_metric.clone().get().clone(),
            distance_metric: dense_index.distance_metric.clone().get().clone(),
            storage_type: dense_index.storage_type.clone().get().clone(),
            size: 0,
            lower_bound: None,
            upper_bound: None,
        };
        Ok(dense_index_data)
    }
}

// TODO use lmdb_init_db function inside this function
pub fn lmdb_init_collections_db(env: &Environment) -> lmdb::Result<Database> {
    env.create_db(Some("collections"), DatabaseFlags::empty())
}

pub fn lmdb_init_db(env: &Environment, name: &str) -> lmdb::Result<Database> {
    env.create_db(Some(name), DatabaseFlags::empty())
}

pub(crate) fn load_collections(env: &Environment, db: Database) -> lmdb::Result<Vec<Collection>> {
    let mut collections = Vec::new();
    let txn = env.begin_ro_txn().unwrap();
    let mut cursor = txn.open_ro_cursor(db).unwrap();
    for (_k, v) in cursor.iter() {
        let col: Collection = from_slice(&v[..]).unwrap();
        collections.push(col);
    }
    Ok(collections)
}

pub fn load_dense_index_data(
    env: &Environment,
    db: Database,
    collection_id: &[u8; 8],
) -> lmdb::Result<DenseIndexData> {
    let txn = env.begin_ro_txn().unwrap();
    let index = txn.get(db, collection_id)?;
    let index: DenseIndexData = from_slice(&index[..]).unwrap();
    Ok(index)
}

pub fn persist_dense_index(
    env: &Environment,
    db: Database,
    dense_index: Arc<DenseIndex>,
) -> Result<(), WaCustomError> {
    let data = DenseIndexData::try_from(dense_index.clone())?;

    // Compute SipHash of the collection name
    // TODO instead use the Collection::get_key() method here
    let mut hasher = SipHasher24::new();
    hasher.write(data.name.as_bytes());
    let hash = hasher.finish();

    let key = hash.to_le_bytes();
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

pub fn delete_dense_index(
    env: &Environment,
    db: Database,
    dense_index: Arc<DenseIndex>,
) -> lmdb::Result<Arc<DenseIndex>> {
    // Compute SipHash of the collection name
    // TODO use the Collection::get_key() method here
    let mut hasher = SipHasher24::new();
    hasher.write(dense_index.database_name.as_bytes());
    let hash = hasher.finish();
    let key = hash.to_le_bytes();
    let mut txn = env.begin_rw_txn()?;
    txn.del(db, &key, None)?;
    txn.commit()?;
    Ok(dense_index)
}
