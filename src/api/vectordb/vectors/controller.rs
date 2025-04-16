use actix_web::{web, HttpResponse, Result};
use std::sync::atomic::Ordering;

use super::{
    dtos::{CreateVectorDto, UpdateVectorDto},
    error::VectorsError,
    service,
};

use crate::models::collection_cache::CollectionCacheExt;
use crate::{
    api_service::run_upload_dense_vectors,
    app_context::AppContext,
    models::{
        common::WaCustomError,
        rpc::{RPCResponseBody, UpsertVectors},
        types::VectorId,
    },
};
pub(crate) async fn create_vector(
    collection_id: web::Path<String>,
    web::Json(create_vector_dto): web::Json<CreateVectorDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    ctx.update_collection_for_transaction(&collection_id)
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("Cache error: {}", e)))?;
    service::create_vector(ctx.into_inner(), &collection_id, create_vector_dto).await?;
    Ok(HttpResponse::Ok().finish())
}

pub(crate) async fn get_vector_by_id(
    path: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let (collection_id, vector_id) = path.into_inner();

    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("Cache error: {}", e)))?;

    let vector =
        service::get_vector_by_id(ctx.into_inner(), &collection_id, VectorId(vector_id)).await?;
    Ok(HttpResponse::Ok().json(vector))
}

pub(crate) async fn update_vector_by_id(
    path: web::Path<(String, VectorId)>,
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

pub(crate) async fn delete_vector_by_id(
    path: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let (collection_id, vector_id) = path.into_inner();
    service::delete_vector_by_id(ctx.into_inner(), &collection_id, vector_id).await?;
    Ok(HttpResponse::NoContent().finish())
}

pub(crate) async fn upsert_vectors(
    collection_id: web::Path<String>,
    web::Json(body): web::Json<UpsertVectors>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VectorsError> {
    let collection_name = collection_id.into_inner();

    ctx.update_collection_for_transaction(&collection_name)
        .map_err(|e| {
            VectorsError::WaCustom(WaCustomError::DatabaseError(format!("Cache error: {}", e)))
        })?;

    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(&collection_name)
        .ok_or_else(|| VectorsError::IndexNotFound)?;

    if !hnsw_index
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return Err(VectorsError::WaCustom(WaCustomError::LockError(
            "Cannot upsert vectors while a transaction is open".to_string(),
        )));
    }

    let vecs_to_upload: Vec<(VectorId, Vec<f32>, Option<crate::metadata::MetadataFields>)> = body
        .vectors
        .into_iter()
        .map(|dense_vec| (dense_vec.id, dense_vec.values, dense_vec.metadata))
        .collect();

    run_upload_dense_vectors(ctx.into_inner(), hnsw_index.clone(), vecs_to_upload)
        .map_err(VectorsError::WaCustom)?;

    Ok(HttpResponse::Ok().json(RPCResponseBody::RespUpsertVectors { insert_stats: None }))
}

pub(crate) async fn check_vector_existence(
    path: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VectorsError> {
    let (collection_id, vector_id_u64) = path.into_inner();

    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| {
            VectorsError::WaCustom(WaCustomError::DatabaseError(format!("Cache error: {}", e)))
        })?;

    let exists =
        service::check_vector_existence(ctx.into_inner(), &collection_id, vector_id_u64).await?;

    if exists {
        // Return 200 OK for HEAD if resource exists
        Ok(HttpResponse::Ok().finish())
    } else {
        // Return 404 Not Found for HEAD if resource doesn't exist
        Ok(HttpResponse::NotFound().finish())
    }
}

pub(crate) async fn fetch_vector_neighbors(
    path: web::Path<(String, u64)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VectorsError> {
    let (collection_id, vector_id_u64) = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| {
            VectorsError::WaCustom(WaCustomError::DatabaseError(format!("Cache error: {}", e)))
        })?;

    let neighbors =
        service::fetch_vector_neighbors(ctx.into_inner(), &collection_id, VectorId(vector_id_u64))
            .await?;
    Ok(HttpResponse::Ok().json(neighbors))
}
