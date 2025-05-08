use actix_web::{web, HttpResponse};

use super::service;
use crate::{
    api::vectordb::transactions::{dtos::UpsertDto, error::TransactionError},
    app_context::AppContext,
    models::collection_cache::CollectionCacheExt,
};

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
