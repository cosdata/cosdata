use actix_web::{web, HttpResponse, Result};

use super::{
    dtos::{CreateVectorDto, FindSimilarVectorsDto, UpdateVectorDto},
    service,
};
use crate::{app_context::AppContext, models::types::VectorId};

pub(crate) async fn create_vector(
    collection_id: web::Path<String>,
    web::Json(create_vector_dto): web::Json<CreateVectorDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let vector =
        service::create_vector(ctx.into_inner(), &collection_id, create_vector_dto).await?;
    Ok(HttpResponse::Ok().json(vector))
}

pub(crate) async fn get_vector_by_id(
    path: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let (collection_id, vector_id) = path.into_inner();
    let vector =
        service::get_vector_by_id(ctx.into_inner(), &collection_id, VectorId(vector_id)).await?;
    Ok(HttpResponse::Ok().json(vector))
}

pub(crate) async fn update_vector_by_id(
    path: web::Path<(String, u64)>,
    web::Json(update_vector_dto): web::Json<UpdateVectorDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let (collection_id, vector_id) = path.into_inner();
    let vector = service::update_vector_by_id(
        ctx.into_inner(),
        &collection_id,
        vector_id,
        update_vector_dto,
    )
    .await?;
    Ok(HttpResponse::Ok().json(vector))
}

pub(crate) async fn find_similar_vectors(
    web::Json(find_similar_vectors): web::Json<FindSimilarVectorsDto>,
) -> Result<HttpResponse> {
    let similar_vectors = service::find_similar_vectors(find_similar_vectors).await?;
    Ok(HttpResponse::Ok().json(similar_vectors))
}

pub(crate) async fn delete_vector_by_id(
    path: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let (collection_id, vector_id) = path.into_inner();
    service::delete_vector_by_id(ctx.into_inner(), &collection_id, vector_id).await?;
    Ok(HttpResponse::NoContent().finish())
}
