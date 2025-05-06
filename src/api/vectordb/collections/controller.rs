use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateCollectionDto},
    service,
};
use crate::api::vectordb::collections::error::CollectionsError;

pub(crate) async fn create_collection(
    web::Json(create_collection_dto): web::Json<CreateCollectionDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let create_collection_response_dto =
        service::create_collection(ctx.into_inner(), create_collection_dto).await?;

    Ok(HttpResponse::Created().json(create_collection_response_dto))
}


pub(crate) async fn get_collection_by_id(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection = service::get_collection_by_id(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(&collection.meta))
}

pub(crate) async fn get_collection_indexing_status(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let status = service::get_collection_indexing_status(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(status))
}

pub(crate) async fn delete_collection_by_id(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    service::delete_collection_by_id(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::NoContent().finish())
}

pub(crate) async fn load_collection(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection = service::load_collection(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(&collection.meta))
}

pub(crate) async fn unload_collection(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    service::unload_collection(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(format!(
        "Collection '{}' successfully unloaded",
        collection_id
    )))
}

pub(crate) async fn get_loaded_collections(ctx: web::Data<AppContext>) -> Result<HttpResponse> {
    let collections = service::get_loaded_collections(ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(collections))
}

pub(crate) async fn list_collections(
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, CollectionsError> {
    let response_dto = service::list_collections(ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(response_dto))
}
