use crate::macros::key;
use crate::models::common::*;
use crate::models::types::*;
use crate::models::versioning::*;
use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use serde_cbor::from_slice;

use super::collection::CollectionMetadata;

/// updates the current version of a collection
pub fn update_current_version(lmdb: &MetaDb, version_hash: Hash) -> Result<(), WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let key = key!(m:current_version);
    let bytes = version_hash.to_le_bytes();

    txn.put(*db, &key, &bytes, WriteFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(())
}

/// updates the current version of a collection
pub fn update_background_version(lmdb: &MetaDb, version_hash: Hash) -> Result<(), WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env
        .begin_rw_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

    let key = key!(m:background_version);
    let bytes = version_hash.to_le_bytes();

    txn.put(*db, &key, &bytes, WriteFlags::empty())
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to put data: {}", e)))?;

    txn.commit().map_err(|e| {
        WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
    })?;

    Ok(())
}

pub fn store_values_range(lmdb: &MetaDb, range: (f32, f32)) -> lmdb::Result<()> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env.begin_rw_txn()?;
    let key = key!(m:values_range);
    let mut bytes = Vec::with_capacity(8);
    bytes.extend(range.0.to_le_bytes());
    bytes.extend(range.1.to_le_bytes());

    txn.put(*db, &key, &bytes, WriteFlags::empty())?;
    txn.commit()?;
    Ok(())
}

pub fn store_values_upper_bound(lmdb: &MetaDb, bound: f32) -> lmdb::Result<()> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env.begin_rw_txn()?;
    let key = key!(m:values_upper_bound);
    let bytes = bound.to_le_bytes();

    txn.put(*db, &key, &bytes, WriteFlags::empty())?;
    txn.commit()?;
    Ok(())
}

pub fn store_average_document_length(lmdb: &MetaDb, len: f32) -> lmdb::Result<()> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env.begin_rw_txn()?;
    let key = key!(m:average_document_length);
    let bytes = len.to_le_bytes();

    txn.put(*db, &key, &bytes, WriteFlags::empty())?;
    txn.commit()?;
    Ok(())
}

pub fn store_highest_internal_id(lmdb: &MetaDb, id: u32) -> lmdb::Result<()> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();

    let mut txn = env.begin_rw_txn()?;
    let key = key!(m:highest_internal_id);
    let bytes = id.to_le_bytes();

    txn.put(*db, &key, &bytes, WriteFlags::empty())?;
    txn.commit()?;
    Ok(())
}

/// retrieves the current version of a collection
pub fn retrieve_current_version(lmdb: &MetaDb) -> Result<Hash, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;
    let current_version_key = key!(m:current_version);

    let serialized_hash = txn.get(*db, &current_version_key).map_err(|e| match e {
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

/// retrieves the current version of a collection
pub fn retrieve_background_version(lmdb: &MetaDb) -> Result<Hash, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;
    let current_version_key = key!(m:background_version);

    let serialized_hash = txn.get(*db, &current_version_key).map_err(|e| match e {
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

pub fn retrieve_values_range(lmdb: &MetaDb) -> Result<Option<(f32, f32)>, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;
    let key = key!(m:values_range);

    let serialized_hash = match txn.get(*db, &key) {
        Ok(bytes) => bytes,
        Err(lmdb::Error::NotFound) => return Ok(None),
        Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
    };

    let bytes: [u8; 8] = serialized_hash.try_into().map_err(|_| {
        WaCustomError::DeserializationError(
            "Failed to deserialize values range: length mismatch".to_string(),
        )
    })?;
    let start = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    let end = f32::from_le_bytes(bytes[4..].try_into().unwrap());

    Ok(Some((start, end)))
}

pub fn retrieve_values_upper_bound(lmdb: &MetaDb) -> Result<Option<f32>, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;
    let key = key!(m:values_upper_bound);

    let serialized = match txn.get(*db, &key) {
        Ok(bytes) => bytes,
        Err(lmdb::Error::NotFound) => return Ok(None),
        Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
    };

    let bytes: [u8; 4] = serialized.try_into().map_err(|_| {
        WaCustomError::DeserializationError(
            "Failed to deserialize values upper bound: length mismatch".to_string(),
        )
    })?;
    let bound = f32::from_le_bytes(bytes);

    Ok(Some(bound))
}

pub fn retrieve_average_document_length(lmdb: &MetaDb) -> Result<Option<f32>, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;
    let key = key!(m:average_document_length);

    let serialized = match txn.get(*db, &key) {
        Ok(bytes) => bytes,
        Err(lmdb::Error::NotFound) => return Ok(None),
        Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
    };

    let bytes: [u8; 4] = serialized.try_into().map_err(|_| {
        WaCustomError::DeserializationError(
            "Failed to deserialize average document length: length mismatch".to_string(),
        )
    })?;
    let len = f32::from_le_bytes(bytes);

    Ok(Some(len))
}

pub fn retrieve_highest_internal_id(lmdb: &MetaDb) -> Result<Option<u32>, WaCustomError> {
    let env = lmdb.env.clone();
    let db = lmdb.db.clone();
    let txn = env
        .begin_ro_txn()
        .map_err(|e| WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;
    let key = key!(m:highest_internal_id);

    let serialized = match txn.get(*db, &key) {
        Ok(bytes) => bytes,
        Err(lmdb::Error::NotFound) => return Ok(None),
        Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
    };

    let bytes: [u8; 4] = serialized.try_into().map_err(|_| {
        WaCustomError::DeserializationError(
            "Failed to deserialize average document length: length mismatch".to_string(),
        )
    })?;
    let id = u32::from_le_bytes(bytes);

    Ok(Some(id))
}

// TODO use lmdb_init_db function inside this function
pub fn lmdb_init_collections_db(env: &Environment) -> lmdb::Result<Database> {
    env.create_db(Some("collections"), DatabaseFlags::empty())
}

pub fn lmdb_init_db(env: &Environment, name: &str) -> lmdb::Result<Database> {
    env.create_db(Some(name), DatabaseFlags::empty())
}

pub(crate) fn load_collections(
    env: &Environment,
    db: Database,
) -> lmdb::Result<Vec<CollectionMetadata>> {
    let mut collections = Vec::new();
    let txn = env.begin_ro_txn().unwrap();
    let mut cursor = txn.open_ro_cursor(db).unwrap();
    for (_k, v) in cursor.iter() {
        let col: CollectionMetadata = from_slice(v).unwrap();
        collections.push(col);
    }
    Ok(collections)
}
