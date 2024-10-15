use actix_web::{
    web::{self},
    HttpResponse, Result,
};

use crate::app_context::AppContext;

use super::{
    dtos::{
        CreateCollectionDto, CreateCollectionDtoResponse, FindCollectionDto, GetCollectionsDto,
    },
    service,
};

pub(crate) async fn create_collection(
    web::Json(create_collection_dto): web::Json<CreateCollectionDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let lower_bound = create_collection_dto.min_val;
    let upper_bound = create_collection_dto.max_val;

    let collection = service::create_collection(ctx.into_inner(), create_collection_dto).await?;

    Ok(HttpResponse::Ok().json(CreateCollectionDtoResponse {
        id: collection.database_name.clone(), // will use the vector store name , till it does have a unique id
        dimensions: collection.quant_dim,
        max_val: lower_bound,
        min_val: upper_bound,
        name: collection.database_name.clone(),
    }))
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
    Ok(HttpResponse::Ok().json(FindCollectionDto {
        id: collection.database_name.clone(),
        dimensions: collection.quant_dim,
        vector_db_name: collection.database_name.clone(),
    }))
}

pub(crate) async fn delete_collection_by_id(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection = service::delete_collection_by_id(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(FindCollectionDto {
        id: collection.database_name.clone(),
        dimensions: collection.quant_dim,
        vector_db_name: collection.database_name.clone(),
    }))
}
