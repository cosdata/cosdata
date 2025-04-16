use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;

use super::dtos::{CreateTFIDFIndexDto, IndexType};
use super::error::IndexesError;
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
    Ok(HttpResponse::Created().json(serde_json::json!({})))
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
    Ok(HttpResponse::Created().json(serde_json::json!({})))
}

pub(crate) async fn create_tf_idf_index(
    web::Json(create_index_dto): web::Json<CreateTFIDFIndexDto>,
    ctx: web::Data<AppContext>,
    collection_id: web::Path<String>,
) -> Result<HttpResponse> {
    service::create_tf_idf_index(
        collection_id.into_inner(),
        create_index_dto,
        ctx.into_inner(),
    )
    .await?;
    Ok(HttpResponse::Created().json(serde_json::json!({})))
}

pub(crate) async fn get_index(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, IndexesError> {
    let index_details = service::get_index(collection_id.into_inner(), ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(index_details))
}

pub(crate) async fn delete_index(
    path: web::Path<(String, IndexType)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, IndexesError> {
    let (collection_id, index_type) = path.into_inner();
    service::delete_index(collection_id, index_type, ctx.into_inner()).await?;
    Ok(HttpResponse::NoContent().finish())
}
