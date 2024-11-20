use std::sync::Arc;

use cosdata::models::collection;

use crate::{api_service::{init_dense_index_for_collection, init_inverted_index_for_collection},
    app_context::AppContext,
    indexes::inverted_index::InvertedIndex,
    models::{collection::Collection, types::DenseIndex},};

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
    }: CreateCollectionDto,
) -> Result<Collection, CollectionsError> {
    let env = &ctx.ain_env.persist;
    let collections_db = &ctx.ain_env.collections_map.lmdb_collections_db;

    let collection = Collection::new(
        name,
        description,
        dense_vector,
        sparse_vector,
        metadata_schema,
        config,
    )
    .map_err(|e| CollectionsError::WaCustomError(e))?;

    // adding the created collection into the in-memory map
    ctx.ain_env
        .collections_map
            .insert_collection(Arc::new(collection.clone()))
        .map_err(|e| CollectionsError::WaCustomError(e))?;

    // persisting collection after creation
    let _ = collection
        .persist(env, collections_db.clone())
        .map_err(|e| CollectionsError::WaCustomError(e));
    Ok(collection)
}

/// creates a dense_index for a collection
pub(crate) async fn create_dense_index(
    ctx: Arc<AppContext>,
    collection: &Collection,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    num_layers: u8,
    auto_config: bool,
) -> Result<Arc<DenseIndex>, CollectionsError> {
    // Call init_vector_store using web::block
    let result = init_dense_index_for_collection(
        ctx,
        collection,
        size,
        lower_bound,
        upper_bound,
        num_layers,
        auto_config,
    )
    .await;
    result.map_err(|e| CollectionsError::FailedToCreateCollection(e.to_string()))
}

/// creates an inverted index for a collection
pub(crate) async fn create_inverted_index(
    ctx: Arc<AppContext>,
    collection: &Collection,
) -> Result<Arc<InvertedIndex>, CollectionsError> {
    let result = init_inverted_index_for_collection(ctx, collection).await;
    result.map_err(|e| CollectionsError::FailedToCreateCollection(e.to_string()))
}

/// gets a list of collections
/// TODO results should be filtered based on search params,
/// if no params provided, it returns all collections
pub(crate) async fn get_collections(
    ctx: Arc<AppContext>,
    get_collections_dto: &GetCollectionsDto,
) -> Result<Vec<GetCollectionsResponseDto>, CollectionsError> {
    let collections = if let Some(name) = &get_collections_dto.name{
        let collection = ctx
            .ain_env
            .collections_map
            .get_collection(&name)
            .ok_or(CollectionsError::NotFound)?;
        vec![GetCollectionsResponseDto {
            name: collection.name.clone(),
            description: collection.description.clone(),
        }]
    } else {
        ctx.ain_env
            .collections_map
            .iter_collections()
            .map(|c| GetCollectionsResponseDto {
                name: c.name.clone(),
                description: c.description.clone(),
            })
            .collect()
    };

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

/// gets a dense index for a collection by name
pub(crate) async fn get_dense_index_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<DenseIndex>, CollectionsError> {
    // Try to get the dense_index from the environment
    let dense_index = match ctx.ain_env.collections_map.get(name) {
        Some(index) => index.clone(),
        None => {
            // dense index not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    Ok(dense_index)
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
        .delete(env, collections_db.clone())
        .map_err(|e| CollectionsError::WaCustomError(e))?;

    // deleting collection from in-memory map
    let collection = ctx
        .ain_env
        .collections_map
        .remove_collection(name)
        .map_err(|e| CollectionsError::WaCustomError(e))?;

    Ok(collection)
}

/// deletes a dense index of a collection by name
pub(crate) async fn delete_dense_index_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<DenseIndex>, CollectionsError> {
    // Try to get the dense index from the environment
    let result = ctx
        .ain_env
        .collections_map
        .remove(name)
        .map_err(CollectionsError::WaCustomError)?;
    match result {
        Some((_, index)) => Ok(index),
        None => {
            // dense index not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    }
}
	 