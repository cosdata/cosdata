use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateDenseIndexDto, CreateSparseIndexDto},
    service,
};

pub(crate) async fn create_dense_index(
    web::Json(create_index_dto): web::Json<CreateDenseIndexDto>,
    ctx: web::Data<AppContext>,
    collection_id: web::Path<String>,
) -> Result<HttpResponse> {
    service::create_dense_index(
        collection_id.into_inner(),
        create_index_dto,
        ctx.into_inner(),
    )
    .await?;
    Ok(HttpResponse::Ok().json(serde_json::json!({})))
}

pub(crate) async fn create_sparse_index(
    web::Json(create_index_dto): web::Json<CreateSparseIndexDto>,
    ctx: web::Data<AppContext>,
    collection_id: web::Path<String>,
) -> Result<HttpResponse> {
    service::create_sparse_index(
        collection_id.into_inner(),
        create_index_dto,
        ctx.into_inner(),
    )
    .await?;
    Ok(HttpResponse::Ok().json(serde_json::json!({})))
}
