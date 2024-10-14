use std::sync::Arc;

use crate::{
    app_context::AppContext,
    models::{collection::Collection, types::VectorStore},
};

use super::{
    dtos::{
        CreateCollectionDto, CreateCollectionDtoResponse, FindCollectionDto, GetCollectionsDto,
    },
    error::CollectionsError,
    repo,
};

pub(crate) async fn create_collection(
    ctx: Arc<AppContext>,
    create_collection_dto: CreateCollectionDto,
) -> Result<CreateCollectionDtoResponse, CollectionsError> {
    let Collection {
        name,
        description,
        config,
        dense_vector,
        metadata_schema,
        sparse_vector,
    } = &repo::create_collection(ctx.clone(), create_collection_dto).await?;

    if dense_vector.enabled {
        let _ = repo::create_vector_store(
            ctx.clone(),
            name,
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
) -> Result<Vec<FindCollectionDto>, CollectionsError> {
    let collections = repo::get_vector_stores(ctx, get_collections_dto).await?;
    Ok(collections)
}

pub(crate) async fn get_collection_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    let store = repo::get_vector_store_by_name(ctx, collection_id).await?;
    Ok(store)
}

pub(crate) async fn delete_collection_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    let store = repo::delete_vector_store_by_name(ctx, collection_id).await?;
    Ok(store)
}
