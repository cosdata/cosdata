use std::sync::Arc;
use async_std::task::current;

use crate::{
    app_context::AppContext,
    models::{common::WaCustomError, meta_persist::{retrieve_current_version, update_current_version}, types::{get_app_env, MetaDb}, versioning::VersionControl}
};
use super::dtos::{VersionListResponse, VersionMetadata, CurrentVersionResponse};
use super::error::VersionError;

pub(crate) async fn list_versions(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<VersionListResponse, VersionError> {
    let env = ctx.ain_env.persist.clone();
    let lmdb = MetaDb::from_env(env.clone(), collection_id)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    let version_control = VersionControl::from_existing(env.clone(), lmdb.db.clone());
    let versions = version_control.get_branch_versions("main")
    .map_err(|e|WaCustomError::DatabaseError(e.to_string()))?;
    let current_hash = retrieve_current_version(&lmdb)
        .map_err(|e| VersionError::DatabaseError(e.to_string()))?;
    let mut versions = versions.into_iter().map(|(hash, version_hash)|{
        VersionMetadata{
            hash,
            version_number: *version_hash.version,
            timestamp: version_hash.timestamp.0,
            vector_count: 0
        }
    }).collect::<Vec<VersionMetadata>>();
    versions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(VersionListResponse{
            versions,
            current_hash
    })
   
}

pub(crate) async fn get_current_version(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CurrentVersionResponse, VersionError> {
    let env = ctx.ain_env.persist.clone();
    let lmdb = MetaDb::from_env(env.clone(), collection_id)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    let version_control = VersionControl::from_existing(env.clone(), lmdb.db.clone());
    let versions = version_control.get_branch_versions("main")
    .map_err(|e|WaCustomError::DatabaseError(e.to_string()))?;
    let current_hash = retrieve_current_version(&lmdb)
        .map_err(|e| VersionError::DatabaseError(e.to_string()))?;
    let current_version = versions.into_iter()
        .find(|(hash, _)| *hash == current_hash)
        .map(|(hash, version_hash)| CurrentVersionResponse {
            hash,
            version_number: *version_hash.version,
            timestamp: version_hash.timestamp.0 as u64,
            vector_count: 0
        })
        .ok_or(VersionError::InvalidVersionHash)?;
    Ok(current_version)
}

pub(crate) async fn set_current_version(
    ctx: Arc<AppContext>,
    collection_id: &str,
    version_hash: &str,
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
