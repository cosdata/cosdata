// use async_std::task::current;
use std::sync::Arc;

use super::dtos::{CurrentVersionResponse, VersionListResponse, VersionMetadata};
use super::error::VersionError;
use crate::{
    app_context::AppContext,
    models::{
        collection_transaction::TransactionStatus, common::WaCustomError,
        meta_persist::retrieve_current_version, types::MetaDb, versioning::VersionControl,
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

    // Calculate cumulative vector counts for each version
    let mut cumulative_count = 0u64;
    let versions = versions
        .into_iter()
        .map(|meta| {
            // Get vector count from transaction status for this version
            let version_records =
                if let Some(status) = collection.transaction_status_map.get_latest(&meta.version) {
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

            // Add this version's records to the cumulative total
            cumulative_count += version_records;

            VersionMetadata {
                version_number: meta.version,
                vector_count: cumulative_count,
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
    // This should be the cumulative total up to the current version
    let vector_count = if let Some(indexing_status) = collection.indexing_status().ok() {
        indexing_status
            .status_summary
            .total_records_indexed_completed
    } else {
        // Fallback: calculate cumulative count manually
        let mut cumulative_count = 0u64;
        let versions = version_control.get_versions().unwrap_or_default();
        for version_meta in versions {
            if *version_meta.version > *version_number {
                break; // Only count versions up to current version
            }
            if let Some(status) = collection
                .transaction_status_map
                .get_latest(&version_meta.version)
            {
                let status = status.read();
                match &*status {
                    TransactionStatus::Complete { summary, .. } => {
                        cumulative_count += summary.total_records_indexed as u64;
                    }
                    _ => {} // Only count completed transactions for total
                }
            }
        }
        cumulative_count
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
