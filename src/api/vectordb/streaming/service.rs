use std::sync::Arc;

use super::repo;
use crate::{
    api::vectordb::{transactions::error::TransactionError, vectors::dtos::CreateVectorDto},
    app_context::AppContext,
    models::types::VectorId,
};

pub(crate) async fn upsert_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), TransactionError> {
    repo::upsert_vectors(ctx, collection_id, vectors).await
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<(), TransactionError> {
    repo::delete_vector_by_id(ctx, collection_id, vector_id).await
}
