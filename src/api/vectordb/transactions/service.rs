use std::sync::Arc;

use crate::{
    api::vectordb::vectors::dtos::{
        CreateDenseVectorDto, CreateSparseVectorDto, CreateVectorResponseDto, UpsertDto,
    },
    app_context::AppContext,
    models::{rpc::DenseVector, versioning::Hash},
};

use super::{dtos::CreateTransactionResponseDto, error::TransactionError, repo};

pub(crate) async fn create_dense_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    repo::create_dense_index_transaction(ctx, collection_id).await
}

pub(crate) async fn create_sparse_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CreateTransactionResponseDto, TransactionError> {
    repo::create_sparse_index_transaction(ctx, collection_id).await
}

pub(crate) async fn commit_dense_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    repo::commit_dense_index_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn commit_sparse_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    repo::commit_sparse_index_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    create_vector_dto: CreateDenseVectorDto,
) -> Result<CreateVectorResponseDto, TransactionError> {
    repo::create_vector_in_transaction(ctx, collection_id, transaction_id, create_vector_dto).await
}

pub(crate) async fn abort_dense_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    repo::abort_dense_index_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn abort_sparse_index_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
) -> Result<(), TransactionError> {
    repo::abort_sparse_index_transaction(ctx, collection_id, transaction_id).await
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vector_id: u32,
) -> Result<(), TransactionError> {
    repo::delete_vector_by_id(ctx, collection_id, transaction_id, vector_id).await
}

pub(crate) async fn upsert_dense_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vectors: Vec<DenseVector>,
) -> Result<(), TransactionError> {
    repo::upsert_dense_vectors(ctx, collection_id, transaction_id, vectors).await
}

pub(crate) async fn upsert_sparse_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction_id: Hash,
    vectors: Vec<CreateSparseVectorDto>,
) -> Result<(), TransactionError> {
    repo::upsert_sparse_vectors(ctx, collection_id, transaction_id, vectors).await
}
