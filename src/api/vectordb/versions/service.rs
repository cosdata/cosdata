// use async_std::task::current;
use std::sync::Arc;

use super::dtos::{CurrentVersionResponse, VersionListResponse, VersionMetadata};
use super::error::VersionError;
use crate::{
    app_context::AppContext,
    models::{
        collection_transaction::TransactionStatus,
        common::WaCustomError, 
        meta_persist::retrieve_current_version, 
        types::MetaDb,
        versioning::VersionControl,
    },
};

pub(crate) async fn list_versions(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<VersionListResponse, VersionError> {
    let env = ctx.ain_env.persist.clone();
    let lmdb = MetaDb::from_env(env.clone(), collection_id)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    let version_control = VersionControl::from_existing(env.clone(), lmdb.db);
    let versions = version_control
        .get_versions()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    let current_version =
        retrieve_current_version(&lmdb).map_err(|e| VersionError::DatabaseError(e.to_string()))?;
    
    // Get the collection to access transaction status map
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| VersionError::DatabaseError("Collection not found".to_string()))?;
    
    let versions = versions
        .into_iter()
        .map(|meta| {
            // Get vector count from transaction status for this version
            let vector_count = if let Some(status) = collection
                .transaction_status_map
                .get_latest(&meta.version)
            {
                let status = status.read();
                match &*status {
                    TransactionStatus::InProgress { progress, .. } => {
                        progress.records_indexed as u64
                    }
                    TransactionStatus::Complete { summary, .. } => {
                        summary.total_records_indexed as u64
                    }
                    TransactionStatus::NotStarted { .. } => 0,
                }
            } else {
                0
            };
            
            VersionMetadata {
                version_number: meta.version,
                vector_count,
            }
        })
        .collect::<Vec<VersionMetadata>>();
    Ok(VersionListResponse {
        versions,
        current_version,
    })
}

pub(crate) async fn get_current_version(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CurrentVersionResponse, VersionError> {
    let env = ctx.ain_env.persist.clone();
    let lmdb = MetaDb::from_env(env.clone(), collection_id)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    let version_control = VersionControl::from_existing(env.clone(), lmdb.db);
    let version_number = version_control
        .get_current_version()
        .map_err(|e| VersionError::DatabaseError(e.to_string()))?;

    // Get the collection to access transaction status map
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| VersionError::DatabaseError("Collection not found".to_string()))?;
    
    // Get vector count from transaction status for the current version
    let vector_count = if let Some(status) = collection
        .transaction_status_map
        .get_latest(&version_number)
    {
        let status = status.read();
        match &*status {
            TransactionStatus::InProgress { progress, .. } => {
                progress.records_indexed as u64
            }
            TransactionStatus::Complete { summary, .. } => {
                summary.total_records_indexed as u64
            }
            TransactionStatus::NotStarted { .. } => 0,
        }
    } else {
        0
    };

    Ok(CurrentVersionResponse {
        version_number,
        vector_count,
    })
}

#[allow(unused)]
pub(crate) async fn set_current_version(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _version_hash: &str,
) -> Result<(), VersionError> {
    // let collection = ctx.ain_env.collections_map.get(collection_id)
    //     .ok_or(VersionError::CollectionNotFound)?;

    // let hash_value = u32::from_str_radix(version_hash, 16)
    //     .map_err(|_| VersionError::InvalidVersionHash)?;
    // let hash = Hash::from(hash_value);

    // update_current_version(&collection.lmdb, hash)
    //     .map_err(|e| VersionError::UpdateFailed(e.to_string()))?;

    // Ok(())
    todo!()
}
