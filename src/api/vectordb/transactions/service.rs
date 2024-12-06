use std::sync::Arc;

use crate::{
    api::vectordb::vectors::dtos::{CreateVectorDto, CreateVectorResponseDto, UpsertDto},
    app_context::AppContext,
    models::versioning::Hash,
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
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    repo::commit_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, TransactionError> {
    repo::create_vector_in_transaction(ctx, collection_id, transaction_id, create_vector_dto).await
}

pub(crate) async fn abort_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    repo::abort_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vector_id: u32,
) -> Result<(), TransactionError> {
    repo::delete_vector_by_id(ctx, collection_id, transaction_id, vector_id).await
}

pub(crate) async fn upsert(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    upsert_dto: UpsertDto,
) -> Result<(), TransactionError> {
    repo::upsert(ctx, collection_id, transaction_id, upsert_dto).await
}
