use std::sync::Arc;

use crate::{
    app_context::AppContext,
    models::{collection::Collection, common::WaCustomError},
};

use super::{
    dtos::{
        CollectionSummaryDto, CreateCollectionDto, GetCollectionsDto, GetCollectionsResponseDto,
        ListCollectionsResponseDto,
    },
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
    }: CreateCollectionDto,
) -> Result<Collection, CollectionsError> {
    let env = &ctx.ain_env.persist;
    let collections_db = &ctx.ain_env.collections_map.lmdb_collections_db;

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
    )
    .map_err(CollectionsError::WaCustomError)?;

    // adding the created collection into the in-memory map
    ctx.ain_env
        .collections_map
        .insert_collection(Arc::new(collection.clone()))
        .map_err(CollectionsError::WaCustomError)?;

    // persisting collection after creation
    let _ = collection
        .persist(env, *collections_db)
        .map_err(CollectionsError::WaCustomError);
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
            name: c.name.clone(),
            description: c.description.clone(),
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

pub(crate) async fn list_collections(
    ctx: Arc<AppContext>,
) -> Result<ListCollectionsResponseDto, CollectionsError> {
    // Iterate over collections stored in the AppContext map
    let summaries: Vec<CollectionSummaryDto> = ctx
        .ain_env
        .collections_map
        .iter_collections()
        .map(|collection_arc| CollectionSummaryDto {
            name: collection_arc.name.clone(),
            description: collection_arc.description.clone(),
        })
        .collect();

    Ok(ListCollectionsResponseDto {
        collections: summaries,
    })
}
