use crate::models::common::*;
use crate::models::types::*;
use crate::models::versioning::*;
use lmdb::{Database, DatabaseFlags, Environment, Error as LmdbError, Transaction, WriteFlags};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub fn store_current_version(
    vec_store: Arc<VectorStore>,
    branch: String,
    version: u32,
) -> Result<VersionHash, WaCustomError> {
    let mut hasher = VersionHasher::new();
    // Generate hashes for main branch
    let hash = hasher.generate_hash(&branch, version, None, None);
    let env = vec_store.version_lmdb.env.clone();
    let db = vec_store.version_lmdb.db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::LmdbError(format!("Failed to begin transaction: {}", e)))?;

    let serialized = serde_cbor::to_vec(&hash)
        .map_err(|e| WaCustomError::SerializationError(format!("Failed to serialize: {}", e)))?;

    txn.put(
        *db.as_ref(),
        &"current_version".to_string(),
        &serialized,
        WriteFlags::empty(),
    )
    .map_err(|e| WaCustomError::LmdbError(format!("Failed to put data: {}", e)))?;

    txn.commit()
        .map_err(|e| WaCustomError::LmdbError(format!("Failed to commit transaction: {}", e)))?;

    Ok(hash)
}

pub fn retrieve_current_version(vec_store: Arc<VectorStore>) -> Result<VersionHash, WaCustomError> {
    let env = vec_store.version_lmdb.env.clone();
    let db = vec_store.version_lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::LmdbError(format!("Failed to begin transaction: {}", e)))?;

    let serialized_hash = txn
        .get(*db.as_ref(), &"current_version".to_string())
        .map_err(|e| match e {
            lmdb::Error::NotFound => {
                WaCustomError::LmdbError(format!("Record not found: {}", "current_version"))
            }
            _ => WaCustomError::LmdbError(e.to_string()),
        })?;

    // Deserialize the CBOR bytes into VersionHash
    let version_hash: VersionHash = serde_cbor::from_slice(serialized_hash).map_err(|e| {
        WaCustomError::DeserializationError(format!("Failed to deserialize VersionHash: {}", e))
    })?;

    Ok(version_hash)
}
