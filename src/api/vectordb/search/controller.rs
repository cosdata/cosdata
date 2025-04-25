use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;
use crate::models::collection_cache::CollectionCacheExt;

use super::dtos::{
    BatchDenseSearchRequestDto, BatchSearchTFIDFDocumentsDto, BatchSparseSearchRequestDto,
    DenseSearchRequestDto, FindSimilarTFIDFDocumentDto, HybridSearchRequestDto,
    SparseSearchRequestDto,
};
use super::error::SearchError;

use super::service;

pub(crate) async fn dense_search(
    path: web::Path<String>,
    web::Json(body): web::Json<DenseSearchRequestDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();

    // Update cache usage
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::dense_search(ctx.into_inner(), &collection_id, body).await?;
    Ok(HttpResponse::Ok().json(results))
}

// Route: `POST /collections/{collection_id}/vectors/search/batch-dense`
pub(crate) async fn batch_dense_search(
    path: web::Path<String>,
    web::Json(body): web::Json<BatchDenseSearchRequestDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();

    // Update cache usage
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::batch_dense_search(ctx.into_inner(), &collection_id, body).await?;

    Ok(HttpResponse::Ok().json(results))
}

pub(crate) async fn sparse_search(
    path: web::Path<String>,
    web::Json(body): web::Json<SparseSearchRequestDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::sparse_search(ctx.into_inner(), &collection_id, body).await?;
    Ok(HttpResponse::Ok().json(results))
}

pub(crate) async fn batch_sparse_search(
    path: web::Path<String>,
    web::Json(body): web::Json<BatchSparseSearchRequestDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::batch_sparse_search(ctx.into_inner(), &collection_id, body).await?;
    Ok(HttpResponse::Ok().json(results))
}

pub(crate) async fn hybrid_search(
    path: web::Path<String>,
    web::Json(body): web::Json<HybridSearchRequestDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::hybrid_search(ctx.into_inner(), &collection_id, body).await?;
    Ok(HttpResponse::Ok().json(results))
}

pub(crate) async fn tf_idf_search(
    path: web::Path<String>,
    web::Json(body): web::Json<FindSimilarTFIDFDocumentDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::tf_idf_search(ctx.into_inner(), &collection_id, body).await?;
    Ok(HttpResponse::Ok().json(results))
}

pub(crate) async fn batch_tf_idf_search(
    path: web::Path<String>,
    web::Json(body): web::Json<BatchSearchTFIDFDocumentsDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::batch_tf_idf_search(ctx.into_inner(), &collection_id, body).await?;
    Ok(HttpResponse::Ok().json(results))
}
