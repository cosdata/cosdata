use actix_web::{web, HttpResponse};

use crate::api::vectordb::vectors::dtos::CreateVectorDto;
use crate::app_context::AppContext;
use crate::models::collection_cache::CollectionCacheExt;
use crate::models::collection_transaction::TransactionStatus;

use super::{dtos::{CreateTransactionResponseDto, UpsertDto}, error::TransactionError, service};

/// Create a new transaction for a collection
///
/// Creates a new transaction for modifying vectors in a collection.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/transactions",
    tag = "transactions",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 200, description = "Transaction created successfully", body = CreateTransactionResponseDto),
        (status = 400, description = "Failed to create transaction"),
        (status = 409, description = "There is an ongoing transaction")
    )
)]
pub(crate) async fn create_transaction(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let collection_id = collection_id.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCreateTransaction(format!("Cache error: {}", e)))?;

    let transaction = service::create_transaction(ctx.into_inner(), &collection_id).await?;

    Ok(HttpResponse::Ok().json(transaction))
}

/// Commit a transaction
///
/// Commits all changes in the transaction to the collection.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/transactions/{transaction_id}/commit",
    tag = "transactions",
    params(
        ("collection_id" = String, Path, description = "Collection identifier"),
        ("transaction_id" = u64, Path, description = "Transaction identifier")
    ),
    responses(
        (status = 204, description = "Transaction committed successfully"),
        (status = 400, description = "Failed to commit transaction")
    )
)]
pub(crate) async fn commit_transaction(
    params: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCommitTransaction(format!("Cache error: {}", e)))?;

    service::commit_transaction(ctx.into_inner(), &collection_id, transaction_id.into()).await?;
    Ok(HttpResponse::NoContent().finish())
}

/// Get transaction status
///
/// Gets the current status of a transaction.
#[utoipa::path(
    get,
    path = "/vectordb/collections/{collection_id}/transactions/{transaction_id}/status",
    tag = "transactions",
    params(
        ("collection_id" = String, Path, description = "Collection identifier"),
        ("transaction_id" = u64, Path, description = "Transaction identifier")
    ),
    responses(
        (status = 200, description = "Transaction status retrieved successfully", body = TransactionStatus),
        (status = 400, description = "Failed to get transaction status"),
        (status = 404, description = "Transaction not found")
    )
)]
pub(crate) async fn get_transaction_status(
    params: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| {
            TransactionError::FailedToGetTransactionStatus(format!("Cache error: {}", e))
        })?;

    let status =
        service::get_transaction_status(ctx.into_inner(), &collection_id, transaction_id.into())
            .await?;

    Ok(HttpResponse::Ok().json(status))
}

/// Create a vector in a transaction
///
/// Creates a new vector as part of an ongoing transaction.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/transactions/{transaction_id}/vectors",
    tag = "transactions",
    params(
        ("collection_id" = String, Path, description = "Collection identifier"),
        ("transaction_id" = u64, Path, description = "Transaction identifier")
    ),
    request_body = CreateVectorDto,
    responses(
        (status = 200, description = "Vector created successfully"),
        (status = 400, description = "Failed to create vector"),
        (status = 404, description = "Transaction not found")
    )
)]
pub(crate) async fn create_vector_in_transaction(
    params: web::Path<(String, u64)>,
    web::Json(create_vector_dto): web::Json<CreateVectorDto>,
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

/// Abort a transaction
///
/// Aborts an ongoing transaction and discards all changes.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/transactions/{transaction_id}/abort",
    tag = "transactions",
    params(
        ("collection_id" = String, Path, description = "Collection identifier"),
        ("transaction_id" = u64, Path, description = "Transaction identifier")
    ),
    responses(
        (status = 204, description = "Transaction aborted successfully"),
        (status = 400, description = "Failed to abort transaction"),
        (status = 404, description = "Transaction not found")
    )
)]
pub(crate) async fn abort_transaction(
    params: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCommitTransaction(format!("Cache error: {}", e)))?;
    service::abort_transaction(ctx.into_inner(), &collection_id, transaction_id.into()).await?;
    Ok(HttpResponse::NoContent().finish())
}

/// Delete a vector in a transaction
///
/// Deletes a vector by its ID as part of an ongoing transaction.
#[utoipa::path(
    delete,
    path = "/vectordb/collections/{collection_id}/transactions/{transaction_id}/vectors/{vector_id}",
    tag = "transactions",
    params(
        ("collection_id" = String, Path, description = "Collection identifier"),
        ("transaction_id" = u64, Path, description = "Transaction identifier"),
        ("vector_id" = String, Path, description = "Vector identifier")
    ),
    responses(
        (status = 204, description = "Vector deleted successfully"),
        (status = 400, description = "Failed to delete vector"),
        (status = 404, description = "Transaction or vector not found")
    )
)]
pub(crate) async fn delete_vector_by_id(
    path: web::Path<(String, u64, String)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id, vector_id) = path.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToDeleteVector(format!("Cache error: {}", e)))?;

    service::delete_vector_by_id(
        ctx.into_inner(),
        &collection_id,
        transaction_id.into(),
        vector_id.into(),
    )
    .await?;
    Ok(HttpResponse::NoContent().finish())
}

/// Upsert vectors in a transaction
///
/// Creates or updates multiple vectors in a single operation as part of an ongoing transaction.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/transactions/{transaction_id}/upsert",
    tag = "transactions",
    params(
        ("collection_id" = String, Path, description = "Collection identifier"),
        ("transaction_id" = u64, Path, description = "Transaction identifier")
    ),
    request_body = UpsertDto,
    responses(
        (status = 200, description = "Vectors upserted successfully"),
        (status = 400, description = "Failed to upsert vectors")
    )
)]
pub(crate) async fn upsert(
    path: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
    web::Json(upsert_dto): web::Json<UpsertDto>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = path.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCreateVector(format!("Cache error: {}", e)))?;

    service::upsert_vectors(
        ctx.into_inner(),
        &collection_id,
        transaction_id.into(),
        upsert_dto.vectors,
    )
    .await?;
    Ok(HttpResponse::Ok().finish())
}
