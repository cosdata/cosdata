// use async_std::task::current;
use std::sync::Arc;

use super::dtos::{CurrentVersionResponse, VersionListResponse, VersionMetadata};
use super::error::VersionError;
use crate::{
    app_context::AppContext,
    models::{
        collection_version_utils::count_live_vectors,
        common::WaCustomError, meta_persist::retrieve_current_version, types::MetaDb,
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
    
    // Get the collection to calculate vector counts
    let collection = ctx.ain_env.collections_map.get_collection(collection_id)
        .ok_or_else(|| VersionError::CollectionNotFound)?;
    
    let versions = versions
        .into_iter()
        .map(|meta| {
            let vector_count = count_live_vectors(&collection, meta.version) as u64;
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

    // Get the collection to calculate vector count
    let collection = ctx.ain_env.collections_map.get_collection(collection_id)
        .ok_or_else(|| VersionError::CollectionNotFound)?;
    
    let vector_count = count_live_vectors(&collection, version_number) as u64;

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
