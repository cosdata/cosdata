use std::sync::Arc;

use crate::models::types::{get_app_env, VectorStore};

use super::error::CollectionsError;

pub(crate) fn get_vector_store_by_name(name: &str) -> Result<Arc<VectorStore>, CollectionsError> {
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
