use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;
use crate::models::collection_cache::CollectionCacheExt;
use crate::models::types::MetricResult;

use super::dtos::{
    BatchDenseSearchRequestDto, BatchSearchResponseDto, BatchSearchTFIDFDocumentsDto,
    BatchSparseSearchRequestDto, DenseSearchRequestDto, FindSimilarTFIDFDocumentDto,
    HybridSearchRequestDto, SearchResponseDto, SearchResultItemDto, SparseSearchRequestDto,
};
use super::error::SearchError;
use crate::api_service::{ann_vector_query, batch_ann_vector_query};

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

    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.clone()))?;

    let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!(
            "Dense (HNSW) index not found for collection '{}'",
            collection_id
        ))
    })?;

    let metadata_filter = match body.filter {
        Some(api_filter) => Some(api_filter),
        None => None,
    };

    let result: Vec<(crate::models::types::VectorId, MetricResult)> = ann_vector_query(
        ctx.into_inner(),
        &collection,
        hnsw_index.clone(),
        body.query_vector,
        metadata_filter,
        body.top_k,
    )
    .await
    .map_err(|e| SearchError::SearchFailed(format!("ANN query failed: {}", e)))?;

    let response_data = SearchResponseDto {
        results: result
            .into_iter()
            .map(|(id, dist)| SearchResultItemDto {
                id,
                score: dist.get_value(),
            })
            .collect(),
    };
    Ok(HttpResponse::Ok().json(response_data))
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

    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.clone()))?;

    let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!(
            "Dense (HNSW) index not found for collection '{}'",
            collection_id
        ))
    })?;

    let metadata_filter = match body.filter {
        Some(api_filter) => Some(api_filter),
        None => None,
    };

    let results: Vec<Vec<(crate::models::types::VectorId, MetricResult)>> = batch_ann_vector_query(
        ctx.into_inner(),
        &collection,
        hnsw_index.clone(),
        body.query_vectors,
        metadata_filter,
        body.top_k,
    )
    .await
    .map_err(|e| SearchError::SearchFailed(format!("Batch ANN query failed: {}", e)))?;

    let response_data: BatchSearchResponseDto = results
        .into_iter()
        .map(|result_list| SearchResponseDto {
            results: result_list
                .into_iter()
                .map(|(id, dist)| SearchResultItemDto {
                    id,
                    score: dist.get_value(),
                })
                .collect(),
        })
        .collect();

    Ok(HttpResponse::Ok().json(response_data))
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
