use actix_web::{web, HttpResponse, Result};

use super::dtos::VectorsQueryDto;
use super::{error::VectorsError, service};

use crate::models::collection_cache::CollectionCacheExt;
use crate::{
    app_context::AppContext,
    models::{common::WaCustomError, types::VectorId},
};

pub(crate) async fn query_vectors(
    collection_id: web::Path<String>,
    web::Query(query): web::Query<VectorsQueryDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection_id = collection_id.into_inner();

    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("Cache error: {}", e)))?;

    let vectors =
        service::query_vectors(ctx.into_inner(), &collection_id, query.document_id).await?;

    Ok(HttpResponse::Ok().json(vectors))
}

pub(crate) async fn get_vector_by_id(
    path: web::Path<(String, String)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let (collection_id, vector_id) = path.into_inner();

    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("Cache error: {}", e)))?;

    let vector =
        service::get_vector_by_id(ctx.into_inner(), &collection_id, VectorId::from(vector_id))
            .await?;
    Ok(HttpResponse::Ok().json(vector))
}

pub(crate) async fn check_vector_existence(
    path: web::Path<(String, String)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VectorsError> {
    let (collection_id, vector_id) = path.into_inner();

    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| {
            VectorsError::WaCustom(WaCustomError::DatabaseError(format!("Cache error: {}", e)))
        })?;

    let exists = service::check_vector_existence(
        ctx.into_inner(),
        &collection_id,
        VectorId::from(vector_id),
    )
    .await?;

    if exists {
        // Return 200 OK for HEAD if resource exists
        Ok(HttpResponse::Ok().finish())
    } else {
        // Return 404 Not Found for HEAD if resource doesn't exist
        Ok(HttpResponse::NotFound().finish())
    }
}

pub(crate) async fn fetch_vector_neighbors(
    path: web::Path<(String, String)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VectorsError> {
    let (collection_id, vector_id_u64) = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| {
            VectorsError::WaCustom(WaCustomError::DatabaseError(format!("Cache error: {}", e)))
        })?;

    let neighbors = service::fetch_vector_neighbors(
        ctx.into_inner(),
        &collection_id,
        VectorId::from(vector_id_u64),
    )
    .await?;
    Ok(HttpResponse::Ok().json(neighbors))
}
