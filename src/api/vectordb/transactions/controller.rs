use actix_web::{web, HttpResponse};

use crate::models::collection_cache::CollectionCacheExt;
use crate::{
    api::vectordb::{indexes::dtos::IndexType, vectors::dtos::CreateDenseVectorDto},
    app_context::AppContext,
};

use super::{
    dtos::{AbortTransactionDto, CommitTransactionDto, CreateTransactionDto, UpsertDto},
    error::TransactionError,
    service,
};

pub(crate) async fn create_transaction(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
    web::Json(CreateTransactionDto { index_type }): web::Json<CreateTransactionDto>,
) -> Result<HttpResponse, TransactionError> {
    let collection_id = collection_id.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCreateTransaction(format!("Cache error: {}", e)))?;

    let transaction = match index_type {
        IndexType::Dense => {
            service::create_dense_index_transaction(ctx.into_inner(), &collection_id).await?
        }
        IndexType::Sparse => {
            service::create_sparse_index_transaction(ctx.into_inner(), &collection_id).await?
        }
    };
    Ok(HttpResponse::Ok().json(transaction))
}

pub(crate) async fn commit_transaction(
    params: web::Path<(String, u32)>,
    ctx: web::Data<AppContext>,
    web::Json(CommitTransactionDto { index_type }): web::Json<CommitTransactionDto>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCommitTransaction(format!("Cache error: {}", e)))?;

    match index_type {
        IndexType::Dense => {
            service::commit_dense_index_transaction(
                ctx.into_inner(),
                &collection_id,
                transaction_id.into(),
            )
            .await?
        }
        IndexType::Sparse => {
            service::commit_sparse_index_transaction(
                ctx.into_inner(),
                &collection_id,
                transaction_id.into(),
            )
            .await?
        }
    };
    Ok(HttpResponse::NoContent().finish())
}

pub(crate) async fn create_vector_in_transaction(
    params: web::Path<(String, u32)>,
    web::Json(create_vector_dto): web::Json<CreateDenseVectorDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();
    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCreateVector(format!("Cache error: {}", e)))?;

    service::create_vector_in_transaction(
        ctx.into_inner(),
        &collection_id,
        transaction_id.into(),
        create_vector_dto,
    )
    .await?;
    Ok(HttpResponse::Ok().finish())
}

pub(crate) async fn abort_transaction(
    params: web::Path<(String, u32)>,
    ctx: web::Data<AppContext>,
    web::Json(AbortTransactionDto { index_type }): web::Json<AbortTransactionDto>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCommitTransaction(format!("Cache error: {}", e)))?;
    match index_type {
        IndexType::Dense => {
            service::abort_dense_index_transaction(
                ctx.into_inner(),
                &collection_id,
                transaction_id.into(),
            )
            .await?
        }
        IndexType::Sparse => {
            service::abort_sparse_index_transaction(
                ctx.into_inner(),
                &collection_id,
                transaction_id.into(),
            )
            .await?
        }
    };
    Ok(HttpResponse::NoContent().finish())
}

pub(crate) async fn delete_vector_by_id(
    path: web::Path<(String, u32, u32)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id, vector_id) = path.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToDeleteVector(format!("Cache error: {}", e)))?;

    service::delete_vector_by_id(
        ctx.into_inner(),
        &collection_id,
        transaction_id.into(),
        vector_id,
    )
    .await?;
    Ok(HttpResponse::NoContent().finish())
}

pub(crate) async fn upsert(
    path: web::Path<(String, u32)>,
    ctx: web::Data<AppContext>,
    web::Json(upsert_dto): web::Json<UpsertDto>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = path.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCreateVector(format!("Cache error: {}", e)))?;

    match upsert_dto {
        UpsertDto::Dense(vectors) => {
            service::upsert_dense_vectors(
                ctx.into_inner(),
                &collection_id,
                transaction_id.into(),
                vectors,
            )
            .await?
        }
        UpsertDto::Sparse(vectors) => {
            service::upsert_sparse_vectors(
                ctx.into_inner(),
                &collection_id,
                transaction_id.into(),
                vectors,
            )
            .await?
        }
        UpsertDto::SparseIdf(documents) => {
            service::upsert_sparse_idf_documents(
                ctx.into_inner(),
                &collection_id,
                transaction_id.into(),
                documents,
            )
            .await?
        }
    };
    Ok(HttpResponse::Ok().finish())
}
