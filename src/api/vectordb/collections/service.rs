use std::sync::Arc;

use crate::{app_context::AppContext, models::types::VectorStore};

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
    // Define the parameters for init_vector_store
    let name = create_collection_dto.name.clone();
    let description = create_collection_dto.description;
    let size = create_collection_dto.dense_vector.dimension as usize;
    let max_cache_level = 5;

    if create_collection_dto.dense_vector.enabled {
        let _ = repo::create_vector_store(ctx, name, size, None, None, max_cache_level).await?;
    } else {
        let _ = repo::create_inverted_index(
            ctx,
            name,
            description,
            create_collection_dto.sparse_vector.auto_create_index,
            create_collection_dto.metadata_schema,
            create_collection_dto.config.max_vectors,
            create_collection_dto.config.replication_factor,
        )
        .await?;
    }

    Ok(CreateCollectionDtoResponse {
        id: create_collection_dto.name.clone(),
        name: create_collection_dto.name,
        dimensions: size,
    })
}

pub(crate) async fn get_collections(
    get_collections_dto: GetCollectionsDto,
) -> Result<Vec<FindCollectionDto>, CollectionsError> {
    let collections = repo::get_vector_stores(get_collections_dto).await?;
    Ok(collections)
}

pub(crate) async fn get_collection_by_id(
    collection_id: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    let store = repo::get_vector_store_by_name(collection_id).await?;
    Ok(store)
}

pub(crate) async fn delete_collection_by_id(
    collection_id: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    let store = repo::delete_vector_store_by_name(collection_id).await?;
    Ok(store)
}
