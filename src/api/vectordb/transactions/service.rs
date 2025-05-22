use std::sync::Arc;

use crate::{
    api::vectordb::vectors::dtos::CreateVectorDto,
    app_context::AppContext,
    models::{
        collection_transaction::TransactionStatus, types::VectorId, versioning::VersionNumber,
    },
};

use super::{dtos::CreateTransactionResponseDto, error::TransactionError, repo};

pub(crate) async fn create_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    repo::create_transaction(ctx, collection_id).await
}

pub(crate) async fn commit_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
) -> Result<(), TransactionError> {
    repo::commit_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn get_transaction_status(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
) -> Result<TransactionStatus, TransactionError> {
    repo::get_transaction_status(ctx, collection_id, transaction_id).await
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
    create_vector_dto: CreateVectorDto,
) -> Result<(), TransactionError> {
    repo::create_vector_in_transaction(ctx, collection_id, transaction_id, create_vector_dto).await
}

pub(crate) async fn abort_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
) -> Result<(), TransactionError> {
    repo::abort_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
    vector_id: VectorId,
) -> Result<(), TransactionError> {
    repo::delete_vector_by_id(ctx, collection_id, transaction_id, vector_id).await
}

pub(crate) async fn upsert_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: VersionNumber,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), TransactionError> {
    repo::upsert_vectors(ctx, collection_id, transaction_id, vectors).await
}
