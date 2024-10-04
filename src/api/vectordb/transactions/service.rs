use std::sync::Arc;

use crate::{
    api::vectordb::vectors::{
        self,
        dtos::{CreateVectorDto, CreateVectorResponseDto},
    },
    app_context::AppContext,
};

use super::{dtos::CreateTransactionResponseDto, error::TransactionError, repo};

pub(crate) async fn create_transaction(
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    repo::create_transaction(collection_id).await
}

pub(crate) async fn commit_transaction(
    collection_id: &str,
    transaction_id: &str,
) -> Result<(), TransactionError> {
    repo::commit_transaction(collection_id, transaction_id).await
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, TransactionError> {
    let vector =
        vectors::service::create_vector_without_committing(ctx, collection_id, create_vector_dto)
            .await
            .map_err(|e| TransactionError::FailedToCreateVector(e.to_string()))?;
    Ok(vector)
}
