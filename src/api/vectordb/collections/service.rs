use std::sync::Arc;

use cosdata::models::types::VectorStore;

use super::{error::CollectionsError, repo};

pub(crate) fn get_collection_by_id(
    collection_id: &str,
) -> Result<Arc<VectorStore>, CollectionsError> {
    let store = repo::get_vector_store_by_name(collection_id)?;
    Ok(store)
}
