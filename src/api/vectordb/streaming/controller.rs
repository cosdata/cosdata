use actix_web::{web, HttpResponse};

use super::service;
use crate::{
    api::vectordb::transactions::{dtos::UpsertDto, error::TransactionError},
    app_context::AppContext,
    models::collection_cache::CollectionCacheExt,
};

/// Upsert vectors into a collection with a synchronous transaction
///
/// This API provides a simplified way to upsert vectors without managing transaction lifecycle.
/// A transaction is created, vectors are upserted, and the transaction is committed in a single request.
/// This operation will insert new vectors or update existing ones based on the vector ID.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/streaming/upsert",
    tag = "streaming",
    params(
        ("collection_id" = String, Path, description = "Collection ID")
    ),
    request_body = UpsertDto,
    responses(
        (status = 200, description = "Vectors upserted successfully"),
        (status = 400, description = "Bad request - validation error"),
        (status = 404, description = "Collection not found"),
        (status = 500, description = "Internal server error")
    )
)]
pub(crate) async fn upsert(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
    web::Json(upsert_dto): web::Json<UpsertDto>,
) -> Result<HttpResponse, TransactionError> {
    let collection_id = collection_id.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCreateTransaction(format!("Cache error: {}", e)))?;

    service::upsert_vectors(ctx.into_inner(), &collection_id, upsert_dto.vectors).await?;

    Ok(HttpResponse::Ok().finish())
}

/// Delete a vector by ID using a synchronous transaction
///
/// This API provides a simplified way to delete a vector without managing transaction lifecycle.
/// A transaction is created, vector is deleted, and the transaction is committed in a single request.
#[utoipa::path(
    delete,
    path = "/vectordb/collections/{collection_id}/streaming/vectors/{vector_id}",
    tag = "streaming",
    params(
        ("collection_id" = String, Path, description = "Collection ID"),
        ("vector_id" = String, Path, description = "Vector ID to delete")
    ),
    responses(
        (status = 204, description = "Vector deleted successfully"),
        (status = 404, description = "Collection or vector not found"),
        (status = 500, description = "Internal server error")
    )
)]
pub(crate) async fn delete_vector_by_id(
    path: web::Path<(String, String)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, TransactionError> {
    let (collection_id, vector_id) = path.into_inner();

    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| TransactionError::FailedToCreateTransaction(format!("Cache error: {}", e)))?;

    service::delete_vector_by_id(ctx.into_inner(), &collection_id, vector_id.into()).await?;

    Ok(HttpResponse::NoContent().finish())
}
