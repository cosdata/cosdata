use std::sync::Arc;
use crate::app_context::AppContext;
use crate::models::common::WaCustomError;

use super::dtos::{
    DenseSearchRequestDto,
    BatchDenseSearchRequestDto,
    SparseSearchRequestDto,
    BatchSparseSearchRequestDto,
    HybridSearchRequestDto,
    SearchResponseDto,
    SearchResultItemDto,
    BatchSearchResponseDto,
    BatchSearchSparseIdfDocumentsDto,
    FindSimilarSparseIdfDocumentDto,
};
use super::error::SearchError;
use super::repo;

#[allow(dead_code)]
pub(crate) async fn dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: DenseSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::dense_search(ctx, collection_id, request)
        .await
        .map_err(|e| match e {
            WaCustomError::NotFound(msg) => SearchError::IndexNotFound(msg),
            other => SearchError::SearchFailed(format!("Repo dense search failed: {}", other)),
        })?;

    Ok(SearchResponseDto {
        results: results.into_iter().map(|(id, metric)| SearchResultItemDto {
            id,
            score: metric.get_value(),
        }).collect(),
    })
}

#[allow(dead_code)]
pub(crate) async fn batch_dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchDenseSearchRequestDto,
) -> Result<BatchSearchResponseDto, SearchError> {
     let results_list = repo::batch_dense_search(ctx, collection_id, request)
        .await
         .map_err(|e| match e {
            WaCustomError::NotFound(msg) => SearchError::IndexNotFound(msg),
            other => SearchError::SearchFailed(format!("Repo batch dense search failed: {}", other)),
        })?;

     Ok(results_list.into_iter().map(|results| SearchResponseDto {
         results: results.into_iter().map(|(id, metric)| SearchResultItemDto {
             id,
             score: metric.get_value(),
         }).collect(),
     }).collect())
}

pub(crate) async fn sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: SparseSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::sparse_search(ctx, collection_id, request)
        .await
        .map_err(|e| match e {
            // Map specific WaCustomError variants if needed
            WaCustomError::NotFound(msg) => SearchError::IndexNotFound(msg),
            other => SearchError::SearchFailed(format!("Repo sparse search failed: {}", other)),
        })?;

    Ok(SearchResponseDto {
        results: results.into_iter().map(|(id, metric)| SearchResultItemDto {
            id,
            score: metric.get_value(),
        }).collect(),
    })
}

pub(crate) async fn batch_sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchSparseSearchRequestDto,
) -> Result<BatchSearchResponseDto, SearchError> {
     let results_list = repo::batch_sparse_search(ctx, collection_id, request)
        .await
         .map_err(|e| match e {
            WaCustomError::NotFound(msg) => SearchError::IndexNotFound(msg),
            other => SearchError::SearchFailed(format!("Repo batch sparse search failed: {}", other)),
        })?;

     Ok(results_list.into_iter().map(|results| SearchResponseDto {
         results: results.into_iter().map(|(id, metric)| SearchResultItemDto {
             id,
             score: metric.get_value(),
         }).collect(),
     }).collect())
}


pub(crate) async fn hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: HybridSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::hybrid_search(ctx, collection_id, request).await?;

    Ok(SearchResponseDto {
        results: results.into_iter().map(|(id, score)| SearchResultItemDto {
            id,
            score,
        }).collect(),
    })
}

pub(crate) async fn sparse_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: FindSimilarSparseIdfDocumentDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::sparse_idf_search(ctx, collection_id, request)
        .await
        .map_err(|e| match e { // Basic error mapping
            WaCustomError::NotFound(msg) => SearchError::IndexNotFound(msg),
            other => SearchError::SearchFailed(format!("Repo sparse IDF search failed: {}", other)),
        })?;

    Ok(SearchResponseDto {
        results: results.into_iter().map(|(id, score)| SearchResultItemDto {
            id,
            score, // Use f32 score directly
        }).collect(),
    })
}

pub(crate) async fn batch_sparse_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchSearchSparseIdfDocumentsDto,
) -> Result<BatchSearchResponseDto, SearchError> {
     let results_list = repo::batch_sparse_idf_search(ctx, collection_id, request)
        .await
         .map_err(|e| match e { // Basic error mapping
            WaCustomError::NotFound(msg) => SearchError::IndexNotFound(msg),
            other => SearchError::SearchFailed(format!("Repo batch sparse IDF search failed: {}", other)),
        })?;

     Ok(results_list.into_iter().map(|results| SearchResponseDto {
         results: results.into_iter().map(|(id, score)| SearchResultItemDto {
             id,
             score, // Use f32 score directly
         }).collect(),
     }).collect())
}
