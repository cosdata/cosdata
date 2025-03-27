use std::sync::Arc;

use crate::{app_context::AppContext, indexes::hnsw::HNSWIndex, models::collection::Collection};

use super::{
    dtos::{
        CreateCollectionDto, CreateCollectionDtoResponse, GetCollectionsDto,
        GetCollectionsResponseDto,
    },
    error::CollectionsError,
    repo,
};

pub(crate) async fn create_collection(
    ctx: Arc<AppContext>,
    create_collection_dto: CreateCollectionDto,
) -> Result<CreateCollectionDtoResponse, CollectionsError> {
    let collection = &repo::create_collection(ctx.clone(), create_collection_dto).await?;

    Ok(CreateCollectionDtoResponse {
        id: collection.name.clone(),
        name: collection.name.clone(),
        description: collection.description.clone(),
    })
}

pub(crate) async fn get_collections(
    ctx: Arc<AppContext>,
    get_collections_dto: GetCollectionsDto,
) -> Result<Vec<GetCollectionsResponseDto>, CollectionsError> {
    let collections = repo::get_collections(ctx, get_collections_dto).await?;
    Ok(collections)
}

/// gets a collection by its id
///
/// currently collection_id = collection.name
pub(crate) async fn get_collection_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<Arc<Collection>, CollectionsError> {
    let collection = repo::get_collection_by_name(ctx, collection_id).await?;
    Ok(collection)
}

/// gets dense index by collection id
///
/// currently collection_id = collection.name
pub(crate) async fn get_dense_index_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<Arc<HNSWIndex>, CollectionsError> {
    let index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(CollectionsError::NotFound)?;
    Ok(index)
}

/// deletes a collection by its id
///
/// currently collection_id = collection.name
pub(crate) async fn delete_collection_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<Arc<Collection>, CollectionsError> {
    let collection = repo::delete_collection_by_name(ctx, collection_id).await?;
    Ok(collection)
}

pub(crate) async fn load_collection(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<Collection, CollectionsError> {
    // First check if collection exists by loading its metadata
    let collection = repo::get_collection_by_name(ctx.clone(), collection_id).await?;

    // Then load it into the cache
    ctx.collection_cache_manager
        .load_collection(collection_id)
        .map_err(|e| CollectionsError::ServerError(format!("Failed to load collection: {}", e)))?;

    Ok(collection.as_ref().clone())
}

pub(crate) async fn unload_collection(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<(), CollectionsError> {
    // Check if collection exists
    let _ = repo::get_collection_by_name(ctx.clone(), collection_id).await?;

    // Then unload it from the cache
    ctx.collection_cache_manager
        .unload_collection(collection_id)
        .map_err(|e| {
            CollectionsError::ServerError(format!("Failed to unload collection: {}", e))
        })?;

    Ok(())
}

pub(crate) async fn get_loaded_collections(
    ctx: Arc<AppContext>,
) -> Result<Vec<String>, CollectionsError> {
    // Just return the list of loaded collections directly
    Ok(ctx.collection_cache_manager.get_loaded_collections())
}
