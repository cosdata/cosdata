use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateCollectionDto, GetCollectionsDto},
    service,
};

pub(crate) async fn create_collection(
    web::Json(create_collection_dto): web::Json<CreateCollectionDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let create_collection_response_dto =
        service::create_collection(ctx.into_inner(), create_collection_dto).await?;

    Ok(HttpResponse::Ok().json(create_collection_response_dto))
}

pub(crate) async fn get_collections(
    web::Query(get_collections_dto): web::Query<GetCollectionsDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collections = service::get_collections(ctx.into_inner(), get_collections_dto).await?;
    Ok(HttpResponse::Ok().json(collections))
}

pub(crate) async fn get_collection_by_id(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection = service::get_collection_by_id(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(collection))
}

pub(crate) async fn delete_collection_by_id(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection = service::delete_collection_by_id(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(collection))
}
