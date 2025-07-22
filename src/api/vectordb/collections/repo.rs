use std::sync::Arc;

use crate::{
    app_context::AppContext,
    models::{
        collection::{Collection, CollectionIndexingStatus},
        common::WaCustomError,
        meta_persist::{update_background_version, update_current_version},
        types::MetaDb,
        versioning::VersionControl,
    },
};

use super::{
    dtos::{CreateCollectionDto, GetCollectionsDto, GetCollectionsResponseDto},
    error::CollectionsError,
};

pub(crate) async fn create_collection(
    ctx: Arc<AppContext>,
    CreateCollectionDto {
        name,
        description,
        config,
        dense_vector,
        metadata_schema,
        sparse_vector,
        tf_idf_options,
        store_raw_text,
    }: CreateCollectionDto,
) -> Result<Arc<Collection>, CollectionsError> {
    // Check if collection already exists
    if ctx.ain_env.collections_map.get_collection(&name).is_some() {
        return Err(CollectionsError::AlreadyExists(name));
    }

    let env = &ctx.ain_env.persist;
    let collections_db = &ctx.ain_env.collections_map.lmdb_collections_db;
    let lmdb = MetaDb::from_env(env.clone(), &name)
        .map_err(|err| CollectionsError::WaCustomError(WaCustomError::from(err)))?;
    let (vcs, hash) = VersionControl::new(env.clone(), lmdb.db)
        .map_err(|err| CollectionsError::WaCustomError(WaCustomError::from(err)))?;

    let metadata_schema = match metadata_schema {
        Some(s) => {
            let schema = s
                .try_into()
                .map_err(|e| CollectionsError::WaCustomError(WaCustomError::MetadataError(e)))?;
            Some(schema)
        }
        None => None,
    };

    let collection = Collection::new(
        name,
        description,
        dense_vector,
        sparse_vector,
        tf_idf_options,
        metadata_schema,
        config,
        store_raw_text,
        lmdb,
        hash,
        vcs,
        &ctx,
    )
    .map_err(CollectionsError::WaCustomError)?;

    // adding the created collection into the in-memory map
    ctx.ain_env
        .collections_map
        .insert_collection(collection.clone())
        .map_err(CollectionsError::WaCustomError)?;

    // persisting collection after creation
    collection
        .persist(env, *collections_db)
        .map_err(CollectionsError::WaCustomError)?;

    collection
        .flush()
        .map_err(CollectionsError::WaCustomError)?;
    update_current_version(&collection.lmdb, hash).map_err(CollectionsError::WaCustomError)?;
    update_background_version(&collection.lmdb, hash).map_err(CollectionsError::WaCustomError)?;
    Ok(collection)
}

/// gets a list of collections
/// TODO results should be filtered based on search params,
/// if no params provided, it returns all collections
pub(crate) async fn get_collections(
    ctx: Arc<AppContext>,
    _get_collections_dto: GetCollectionsDto,
) -> Result<Vec<GetCollectionsResponseDto>, CollectionsError> {
    let collections = ctx
        .ain_env
        .collections_map
        .iter_collections()
        .map(|c| GetCollectionsResponseDto {
            name: c.meta.name.clone(),
            description: c.meta.description.clone(),
        })
        .collect();
    Ok(collections)
}

/// gets a collection by its name
pub(crate) async fn get_collection_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<Collection>, CollectionsError> {
    let collection = match ctx.ain_env.collections_map.get_collection(name) {
        Some(collection) => collection.clone(),
        None => {
            // dense index not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    Ok(collection)
}

pub(crate) async fn get_collection_indexing_status(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<CollectionIndexingStatus, CollectionsError> {
    let collection = match ctx.ain_env.collections_map.get_collection(name) {
        Some(collection) => collection.clone(),
        None => {
            // dense index not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    collection
        .indexing_status()
        .map_err(CollectionsError::WaCustomError)
}

pub(crate) async fn delete_collection_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<Collection>, CollectionsError> {
    let env = &ctx.ain_env.persist;
    let collections_db = &ctx.ain_env.collections_map.lmdb_collections_db;

    let collection = get_collection_by_name(ctx.clone(), name).await?;

    // deleting collection from disk
    collection
        .delete(env, *collections_db)
        .map_err(CollectionsError::WaCustomError)?;

    // deleting collection from in-memory map
    let collection = ctx
        .ain_env
        .collections_map
        .remove_collection(name)
        .map_err(CollectionsError::WaCustomError)?;

    Ok(collection)
}
