use std::sync::Arc;

use crate::{app_context::AppContext, models::types::VectorStore};

use super::{
    dtos::{CreateCollectionDto, FindCollectionDto, GetCollectionsDto},
    error::CollectionsError,
    repo,
};

pub(crate) async fn create_collection(
    ctx: Arc<AppContext>,
    create_collection_dto: CreateCollectionDto,
) -> Result<Arc<VectorStore>, CollectionsError> {
    // Define the parameters for init_vector_store
    let name = create_collection_dto.vector_db_name;
    let size = create_collection_dto.dimensions as usize;
    let lower_bound = create_collection_dto.min_val;
    let upper_bound = create_collection_dto.max_val;
    // ---------------------------
    // -- TODO Maximum cache level
    // ---------------------------
    let max_cache_level = 5;

    repo::create_vector_store(ctx, name, size, lower_bound, upper_bound, max_cache_level).await
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
