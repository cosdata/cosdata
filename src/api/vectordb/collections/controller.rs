use actix_web::{
    web::{self},
    HttpResponse, Result,
};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateCollectionDto, FindCollectionDto, GetCollectionsDto},
    service,
};

pub(crate) async fn create_collection(
    web::Json(create_collection_dto): web::Json<CreateCollectionDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let create_collection_response_dto =
        service::create_collection(ctx.into_inner(), &create_collection_dto).await?;

    Ok(HttpResponse::Ok().json(create_collection_response_dto))
}

pub(crate) async fn get_collections(
    web::Query(get_collections_dto): web::Query<GetCollectionsDto>,
) -> Result<HttpResponse> {
    let collections = service::get_collections(get_collections_dto).await?;
    Ok(HttpResponse::Ok().json(collections))
}

pub(crate) async fn get_collection_by_id(collection_id: web::Path<String>) -> Result<HttpResponse> {
    let collection = service::get_collection_by_id(&collection_id).await?;
    Ok(HttpResponse::Ok().json(FindCollectionDto {
        id: collection.database_name.clone(),
        dimensions: collection.quant_dim,
        vector_db_name: collection.database_name.clone(),
    }))
}

pub(crate) async fn delete_collection_by_id(
    collection_id: web::Path<String>,
) -> Result<HttpResponse> {
    let collection = service::delete_collection_by_id(&collection_id).await?;
    Ok(HttpResponse::Ok().json(FindCollectionDto {
        id: collection.database_name.clone(),
        dimensions: collection.quant_dim,
        vector_db_name: collection.database_name.clone(),
    }))
}
