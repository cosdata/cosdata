use std::sync::Arc;

use crate::{
    api_service::{init_inverted_index, init_vector_store},
    app_context::AppContext,
    indexes::inverted_index::InvertedIndex,
    models::types::{get_app_env, Collection, VectorStore},
};

use super::{
    dtos::{CreateCollectionDto, FindCollectionDto, GetCollectionsDto},
    error::CollectionsError,
};

pub(crate) async fn create_collection(
    CreateCollectionDto {
        name,
        description,
        config,
        dense_vector,
        metadata_schema,
        sparse_vector,
    }: &CreateCollectionDto,
) -> Result<Arc<Collection>, CollectionsError> {
    let collection = Collection::new(
        name,
        description,
        dense_vector,
        sparse_vector,
        metadata_schema,
        config,
    );

    Ok(Arc::new(collection))
}

pub(crate) async fn create_vector_store(
    ctx: Arc<AppContext>,
    name: &str,
    size: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    max_cache_level: u8,
) -> Result<Arc<VectorStore>, CollectionsError> {
    // Call init_vector_store using web::block
    let result = init_vector_store(
        ctx,
        name.into(),
        size,
        lower_bound,
        upper_bound,
        max_cache_level,
    )
    .await;
    result.map_err(|e| CollectionsError::FailedToCreateCollection(e.to_string()))
}

pub(crate) async fn create_inverted_index(
    ctx: Arc<AppContext>,
    name: &str,
    description: &Option<String>,
    auto_create_index: bool,
    metadata_schema: &Option<String>,
    max_vectors: Option<i32>,
    replication_factor: Option<i32>,
) -> Result<Arc<InvertedIndex>, CollectionsError> {
    let result = init_inverted_index(
        ctx,
        name.into(),
        description.clone(),
        auto_create_index,
        metadata_schema.clone(),
        max_vectors,
        replication_factor,
    )
    .await;
    result.map_err(|e| CollectionsError::FailedToCreateCollection(e.to_string()))
}

pub(crate) async fn get_vector_stores(
    _get_collections_dto: GetCollectionsDto,
) -> Result<Vec<FindCollectionDto>, CollectionsError> {
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return Err(CollectionsError::FailedToGetAppEnv),
    };
    let vec_store = env
        .vector_store_map
        .iter()
        .map(|v| FindCollectionDto {
            id: v.database_name.clone(),
            dimensions: v.quant_dim,
            vector_db_name: v.database_name.clone(),
        })
        .collect();
    Ok(vec_store)
}

pub(crate) async fn get_vector_store_by_name(
    name: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return Err(CollectionsError::FailedToGetAppEnv),
    };
    // Try to get the vector store from the environment
    let vec_store = match env.vector_store_map.get(name) {
        Some(store) => store.clone(),
        None => {
            // Vector store not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    Ok(vec_store)
}

pub(crate) async fn delete_vector_store_by_name(
    name: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return Err(CollectionsError::FailedToGetAppEnv),
    };
    // Try to get the vector store from the environment
    let vec_store = match env.vector_store_map.remove(name) {
        Some((_, store)) => store,
        None => {
            // Vector store not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    Ok(vec_store)
}
