use actix_web::{web, HttpResponse};

use crate::api::vectordb::vectors::dtos::CreateVectorDto;
use crate::app_context::AppContext;
use crate::models::collection_cache::CollectionCacheExt;

use super::{dtos::UpsertDto, error::TransactionError, service};

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

pub(crate) async fn commit_transaction(
    params: web::Path<(String, u32)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCommitTransaction(format!("Cache error: {}", e)))?;

    service::commit_transaction(ctx.into_inner(), &collection_id, transaction_id.into()).await?;
    Ok(HttpResponse::NoContent().finish())
}

pub(crate) async fn get_transaction_status(
    params: web::Path<(String, u32)>,
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

pub(crate) async fn create_vector_in_transaction(
    params: web::Path<(String, u32)>,
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

pub(crate) async fn abort_transaction(
    params: web::Path<(String, u32)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, transaction_id) = params.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCommitTransaction(format!("Cache error: {}", e)))?;
    service::abort_transaction(ctx.into_inner(), &collection_id, transaction_id.into()).await?;
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

    service::upsert_vectors(
        ctx.into_inner(),
        &collection_id,
        transaction_id.into(),
        upsert_dto.vectors,
    )
    .await?;
    Ok(HttpResponse::Ok().finish())
}
