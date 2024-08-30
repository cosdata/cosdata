use crate::models::common::*;
use crate::models::types::*;
use crate::models::versioning::*;
use lmdb::{Transaction, WriteFlags};
use std::sync::Arc;

pub fn store_current_version(
    lmdb: &MetaDb,
    vcs: Arc<VersionControl>,
    branch: &str,
    version: u32,
) -> Result<Hash, WaCustomError> {
    // Generate hashes for main branch
    let hash = vcs
        .generate_hash(branch, version.into())
        .map_err(|err| WaCustomError::DatabaseError(format!("Unable to generate hash: {}", err)))?;
    let env = lmdb.env.clone();
    let db = lmdb.metadata_db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let bytes = hash.to_le_bytes();

    txn.put(*db, &"current_version", &bytes, WriteFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(hash)
}

pub fn retrieve_current_version(lmdb: &MetaDb) -> Result<Hash, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.metadata_db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let serialized_hash = txn
        .get(*db, &"current_version".to_string())
        .map_err(|e| match e {
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
