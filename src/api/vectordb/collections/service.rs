use std::sync::Arc;

use crate::{
    app_context::AppContext,
    indexes::inverted_index::InvertedIndex,
    models::{collection::Collection, types::DenseIndex},
};

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

    if collection.dense_vector.enabled {
        let _ = repo::create_dense_index(
            ctx.clone(),
            collection,
            collection.dense_vector.dimension as usize,
            None,
            None,
            5,
            true,
        )
        .await?;
    }
    if collection.sparse_vector.enabled {
        let _ = repo::create_inverted_index(ctx, collection).await?;
    }

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
) -> Result<Arc<DenseIndex>, CollectionsError> {
    let index = repo::get_dense_index_by_name(ctx, collection_id).await?;
    Ok(index)
}

/// gets inverted index by collection id
///
/// currently collection_id = collection.name
pub(crate) async fn get_inverted_index_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<Arc<InvertedIndex>, CollectionsError> {
    let index = repo::get_inverted_index_by_name(ctx, collection_id).await?;
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
