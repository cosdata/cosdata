use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;
use crate::models::collection_cache::CollectionCacheExt;

use super::dtos::{
    BatchDenseSearchRequestDto, BatchGeoFenceSearchRequestDto, BatchSearchResponseDto,
    BatchSearchTFIDFDocumentsDto, DenseSearchRequestDto, FindSimilarTFIDFDocumentDto,
    GeoFenceSearchRequestDto, HybridSearchRequestDto, SearchResponseDto,
};
use super::error::SearchError;

use super::service;

/// Search using a dense vector embedding
///
/// Performs a similarity search using dense vector embeddings with optional filtering.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/search/dense",
    tag = "search",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    request_body = DenseSearchRequestDto,
    responses(
        (status = 200, description = "Search successfully completed", body = SearchResponseDto),
        (status = 404, description = "Collection not found", body = String),
        (status = 400, description = "Invalid filter or other request error", body = String),
        (status = 500, description = "Internal server error", body = String)
    )
)]
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

/// Batch search using dense vector embeddings
///
/// Performs multiple similarity searches using dense vector embeddings in a single request.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/search/batch-dense",
    tag = "search",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    request_body = BatchDenseSearchRequestDto,
    responses(
        (status = 200, description = "Batch search successfully completed", body = BatchSearchResponseDto),
        (status = 404, description = "Collection not found", body = String),
        (status = 400, description = "Invalid filter or other request error", body = String),
        (status = 500, description = "Internal server error", body = String)
    )
)]
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

/// Geofence search.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/search/geofence",
    tag = "search",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    request_body = GeoFenceSearchRequestDto,
    responses(
        (status = 200, description = "Search successfully completed", body = SearchResponseDto),
        (status = 404, description = "Collection not found", body = String),
        (status = 400, description = "Invalid request error", body = String),
        (status = 500, description = "Internal server error", body = String)
    )
)]
pub(crate) async fn geofence_search(
    path: web::Path<String>,
    web::Json(body): web::Json<GeoFenceSearchRequestDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::geofence_search(ctx.into_inner(), &collection_id, body).await?;
    Ok(HttpResponse::Ok().json(results))
}

/// Batch geofence search
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/search/batch-geofence",
    tag = "search",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    request_body = BatchGeoFenceSearchRequestDto,
    responses(
        (status = 200, description = "Batch search successfully completed", body = BatchSearchResponseDto),
        (status = 404, description = "Collection not found", body = String),
        (status = 400, description = "Invalid filter or other request error", body = String),
        (status = 500, description = "Internal server error", body = String)
    )
)]
pub(crate) async fn batch_geofence_search(
    path: web::Path<String>,
    web::Json(body): web::Json<BatchGeoFenceSearchRequestDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, SearchError> {
    let collection_id = path.into_inner();

    // Update cache usage
    ctx.update_collection_for_query(&collection_id)
        .map_err(|e| SearchError::InternalServerError(format!("Cache update error: {}", e)))?;

    let results = service::batch_geofence_search(ctx.into_inner(), &collection_id, body).await?;

    Ok(HttpResponse::Ok().json(results))
}

/// Hybrid search combining different search approaches
///
/// Performs a hybrid search combining multiple search strategies (dense/sparse/TFIDF).
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/search/hybrid",
    tag = "search",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    request_body = HybridSearchRequestDto,
    responses(
        (status = 200, description = "Hybrid search successfully completed", body = SearchResponseDto),
        (status = 404, description = "Collection not found", body = String),
        (status = 400, description = "Invalid request error", body = String),
        (status = 500, description = "Internal server error", body = String)
    )
)]
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

/// Search for similar documents using TF-IDF
///
/// Performs a search for similar documents using TF-IDF scoring.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/search/tf-idf",
    tag = "search",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    request_body = FindSimilarTFIDFDocumentDto,
    responses(
        (status = 200, description = "TF-IDF search successfully completed", body = SearchResponseDto),
        (status = 404, description = "Collection not found", body = String),
        (status = 400, description = "Invalid request error", body = String),
        (status = 500, description = "Internal server error", body = String)
    )
)]
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

/// Batch search for similar documents using TF-IDF
///
/// Performs multiple searches for similar documents using TF-IDF scoring in a single request.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/search/batch-tf-idf",
    tag = "search",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    request_body = BatchSearchTFIDFDocumentsDto,
    responses(
        (status = 200, description = "Batch TF-IDF search successfully completed", body = BatchSearchResponseDto),
        (status = 404, description = "Collection not found", body = String),
        (status = 400, description = "Invalid request error", body = String),
        (status = 500, description = "Internal server error", body = String)
    )
)]
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
