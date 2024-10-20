use std::sync::Arc;

use crate::{
    app_context::AppContext,
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

    let Collection {
        name,
        description,
        config,
        dense_vector,
        metadata_schema,
        sparse_vector,
    } = collection;

    if dense_vector.enabled {
        let _ = repo::create_dense_index(
            ctx.clone(),
            collection,
            dense_vector.dimension as usize,
            None,
            None,
            5,
        )
        .await?;
    }
    if sparse_vector.enabled {
        let _ = repo::create_inverted_index(
            ctx,
            name,
            description,
            sparse_vector.auto_create_index,
            metadata_schema,
            config.max_vectors,
            config.replication_factor,
        )
        .await?;
    }

    Ok(CreateCollectionDtoResponse {
        id: name.clone(),
        name: name.clone(),
        description: description.clone(),
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
