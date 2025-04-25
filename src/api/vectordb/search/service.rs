use crate::app_context::AppContext;
use std::sync::Arc;

use super::dtos::{
    BatchDenseSearchRequestDto, BatchSearchResponseDto, BatchSearchTFIDFDocumentsDto,
    BatchSparseSearchRequestDto, DenseSearchRequestDto, FindSimilarTFIDFDocumentDto,
    HybridSearchRequestDto, SearchResponseDto, SearchResultItemDto, SparseSearchRequestDto,
};
use super::error::SearchError;
use super::repo;

pub(crate) async fn dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: DenseSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::dense_search(ctx, collection_id, request).await?;

    Ok(SearchResponseDto {
        results: results
            .into_iter()
            .map(|(id, document_id, score, text)| SearchResultItemDto {
                id,
                document_id,
                score,
                text,
            })
            .collect(),
    })
}

pub(crate) async fn batch_dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchDenseSearchRequestDto,
) -> Result<BatchSearchResponseDto, SearchError> {
    let results_list = repo::batch_dense_search(ctx, collection_id, request).await?;

    Ok(results_list
        .into_iter()
        .map(|results| SearchResponseDto {
            results: results
                .into_iter()
                .map(|(id, document_id, score, text)| SearchResultItemDto {
                    id,
                    document_id,
                    score,
                    text,
                })
                .collect(),
        })
        .collect())
}

pub(crate) async fn sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: SparseSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::sparse_search(ctx, collection_id, request).await?;

    Ok(SearchResponseDto {
        results: results
            .into_iter()
            .map(|(id, document_id, score, text)| SearchResultItemDto {
                id,
                document_id,
                score,
                text,
            })
            .collect(),
    })
}

pub(crate) async fn batch_sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchSparseSearchRequestDto,
) -> Result<BatchSearchResponseDto, SearchError> {
    let results_list = repo::batch_sparse_search(ctx, collection_id, request).await?;

    Ok(results_list
        .into_iter()
        .map(|results| SearchResponseDto {
            results: results
                .into_iter()
                .map(|(id, document_id, score, text)| SearchResultItemDto {
                    id,
                    document_id,
                    score,
                    text,
                })
                .collect(),
        })
        .collect())
}

pub(crate) async fn hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: HybridSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::hybrid_search(ctx, collection_id, request).await?;

    Ok(SearchResponseDto {
        results: results
            .into_iter()
            .map(|(id, document_id, score, text)| SearchResultItemDto {
                id,
                document_id,
                score,
                text,
            })
            .collect(),
    })
}

pub(crate) async fn tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: FindSimilarTFIDFDocumentDto,
) -> Result<SearchResponseDto, SearchError> {
    let results = repo::tf_idf_search(ctx, collection_id, request).await?;

    Ok(SearchResponseDto {
        results: results
            .into_iter()
            .map(|(id, document_id, score, text)| SearchResultItemDto {
                id,
                document_id,
                score,
                text,
            })
            .collect(),
    })
}

pub(crate) async fn batch_tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchSearchTFIDFDocumentsDto,
) -> Result<BatchSearchResponseDto, SearchError> {
    let results_list = repo::batch_tf_idf_search(ctx, collection_id, request).await?;

    Ok(results_list
        .into_iter()
        .map(|results| SearchResponseDto {
            results: results
                .into_iter()
                .map(|(id, document_id, score, text)| SearchResultItemDto {
                    id,
                    document_id,
                    score,
                    text,
                })
                .collect(),
        })
        .collect())
}
