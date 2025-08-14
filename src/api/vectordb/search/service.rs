use crate::app_context::AppContext;
use std::sync::Arc;

use super::dtos::{
    BatchDenseSearchRequestDto, BatchHybridSearchRequestDto, BatchSearchResponseDto,
    BatchSearchTFIDFDocumentsDto, BatchSparseSearchRequestDto, DenseSearchRequestDto,
    FindSimilarTFIDFDocumentDto, HybridSearchRequestDto, SearchResponseDto, SearchResultItemDto,
    SparseSearchRequestDto,
};
use super::error::SearchError;
use super::repo;

pub(crate) async fn dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: DenseSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let (results, warning) = repo::dense_search(ctx, collection_id, request).await?;

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
        warning,
    })
}

pub(crate) async fn batch_dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchDenseSearchRequestDto,
) -> Result<BatchSearchResponseDto, SearchError> {
    let (results_list, warning) = repo::batch_dense_search(ctx, collection_id, request).await?;

    Ok(BatchSearchResponseDto {
        responses: results_list
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
                warning: None,
            })
            .collect(),
        warning,
    })
}

pub(crate) async fn sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: SparseSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let (results, warning) = repo::sparse_search(ctx, collection_id, request).await?;

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
        warning,
    })
}

pub(crate) async fn batch_sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchSparseSearchRequestDto,
) -> Result<BatchSearchResponseDto, SearchError> {
    let (results_list, warning) = repo::batch_sparse_search(ctx, collection_id, request).await?;

    Ok(BatchSearchResponseDto {
        responses: results_list
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
                warning: None,
            })
            .collect(),
        warning,
    })
}

pub(crate) async fn hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: HybridSearchRequestDto,
) -> Result<SearchResponseDto, SearchError> {
    let (results, warning) = repo::hybrid_search(ctx, collection_id, request).await?;

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
        warning,
    })
}

pub(crate) async fn batch_hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchHybridSearchRequestDto,
) -> Result<BatchSearchResponseDto, SearchError> {
    let (results_list, warning) = repo::batch_hybrid_search(ctx, collection_id, request).await?;

    Ok(BatchSearchResponseDto {
        responses: results_list
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
                warning: None,
            })
            .collect(),
        warning,
    })
}

pub(crate) async fn tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: FindSimilarTFIDFDocumentDto,
) -> Result<SearchResponseDto, SearchError> {
    let (results, warning) = repo::tf_idf_search(ctx, collection_id, request).await?;

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
        warning,
    })
}

pub(crate) async fn batch_tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: BatchSearchTFIDFDocumentsDto,
) -> Result<BatchSearchResponseDto, SearchError> {
    let (results_list, warning) = repo::batch_tf_idf_search(ctx, collection_id, request).await?;

    Ok(BatchSearchResponseDto {
        responses: results_list
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
                warning: None,
            })
            .collect(),
        warning,
    })
}
