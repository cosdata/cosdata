use std::sync::Arc;

use crate::{api_service::init_vector_store, app_context::AppContext, models::types::VectorStore};

use super::{
    dtos::{FindCollectionDto, GetCollectionsDto},
    error::CollectionsError,
};

pub(crate) async fn create_vector_store(
    ctx: Arc<AppContext>,
    name: String,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    num_layers: u8,
    auto_config: bool,
) -> Result<Arc<VectorStore>, CollectionsError> {
    // Call init_vector_store using web::block
    let result = init_vector_store(
        ctx,
        name,
        size,
        lower_bound,
        upper_bound,
        num_layers,
        auto_config,
    )
    .await;
    result.map_err(|e| CollectionsError::FailedToCreateCollection(e.to_string()))
}

pub(crate) async fn get_vector_stores(
    ctx: Arc<AppContext>,
    _get_collections_dto: GetCollectionsDto,
) -> Result<Vec<FindCollectionDto>, CollectionsError> {
    let vec_store = ctx
        .ain_env
        .vector_store_map
        .iter()
        .map(|v| FindCollectionDto {
            id: v.database_name.clone(),
            dimensions: v.dim,
            vector_db_name: v.database_name.clone(),
        })
        .collect();
    Ok(vec_store)
}

pub(crate) async fn get_vector_store_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    // Try to get the vector store from the environment
    let vec_store = match ctx.ain_env.vector_store_map.get(name) {
        Some(store) => store.clone(),
        None => {
            // Vector store not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    Ok(vec_store)
}

pub(crate) async fn delete_vector_store_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    // Try to get the vector store from the environment
    let result = ctx
        .ain_env
        .vector_store_map
        .remove(name)
        .map_err(CollectionsError::WaCustomError)?;
    match result {
        Some((_, store)) => Ok(store),
        None => {
            // Vector store not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    }
}
