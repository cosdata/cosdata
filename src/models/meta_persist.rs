use crate::models::common::*;
use crate::models::types::*;
use crate::models::versioning::*;
use crate::quantization::StorageType;
use lmdb::Cursor;
use lmdb::Database;
use lmdb::DatabaseFlags;
use lmdb::Environment;
use lmdb::{Transaction, WriteFlags};
use serde_cbor::from_slice;
use serde_cbor::to_vec;
use siphasher::sip::SipHasher24;
use std::hash::Hasher;
use std::sync::Arc;

use super::lazy_load::FileIndex;

pub fn store_current_version(lmdb: &MetaDb, hash: Hash) -> Result<(), WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let bytes = hash.to_le_bytes();

    txn.put(*db, &"current_version", &bytes, WriteFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(())
}

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
pub struct VecStoreData {
    pub name: String,
    pub max_level: u8,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub quant_dim: usize,
    pub file_index: FileIndex,
    pub quantization_metric: Arc<QuantizationMetric>,
    pub distance_metric: Arc<DistanceMetric>,
    pub storage_type: StorageType,
    pub size: usize,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
}

impl TryFrom<Arc<VectorStore>> for VecStoreData {
    type Error = WaCustomError;
    fn try_from(store: Arc<VectorStore>) -> Result<Self, Self::Error> {
        let offset = store.root_vec_offset().ok_or(WaCustomError::NodeError(
            "FileIndex must be set for root node".to_owned(),
        ))?;
        println!("Offset of root node: {offset:?}");
        if let FileIndex::Invalid = offset {
            return Err(WaCustomError::NodeError(
                "FileIndex must be valid for root node".to_owned(),
            ));
        };
        let res = Self {
            name: store.database_name.clone(),
            max_level: store.max_cache_level,
            levels_prob: store.levels_prob.clone(),
            quant_dim: store.quant_dim,
            file_index: offset,
            quantization_metric: store.quantization_metric.clone(),
            distance_metric: store.distance_metric.clone(),
            storage_type: store.storage_type.clone(),
            size: 0,
            lower_bound: None,
            upper_bound: None,
        };
        Ok(res)
    }
}

pub fn lmdb_init_collections_db(env: &Environment) -> lmdb::Result<Database> {
    env.create_db(Some("collections"), DatabaseFlags::empty())
}

pub fn load_collections(env: &Environment, db: Database) -> lmdb::Result<Vec<VecStoreData>> {
    let mut res = Vec::new();
    let txn = env.begin_ro_txn().unwrap();
    let mut cursor = txn.open_ro_cursor(db).unwrap();
    for (_k, v) in cursor.iter() {
        let val: VecStoreData = from_slice(&v[..]).unwrap();
        res.push(val);
    }
    Ok(res)
}

pub fn persist_vector_store(
    env: &Environment,
    db: Database,
    vec_store: Arc<VectorStore>,
) -> Result<(), WaCustomError> {
    let data = VecStoreData::try_from(vec_store.clone())?;

    // Compute SipHash of the vector_store/collection name
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

pub fn delete_vector_store(
    env: &Environment,
    db: Database,
    vec_store: Arc<VectorStore>,
) -> lmdb::Result<Arc<VectorStore>> {
    // Compute SipHash of the vector_store/collection name
    let mut hasher = SipHasher24::new();
    hasher.write(vec_store.database_name.as_bytes());
    let hash = hasher.finish();
    let key = hash.to_le_bytes();
    let mut txn = env.begin_rw_txn()?;
    txn.del(db, &key, None)?;
    txn.commit()?;
    Ok(vec_store)
}
