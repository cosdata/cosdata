use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::dtos;
use super::error::SearchError;
use crate::app_context::AppContext;
use crate::indexes::hnsw::{DenseSearchInput, DenseSearchOptions};
use crate::indexes::inverted::{SparseSearchInput, SparseSearchOptions};
use crate::indexes::tf_idf::{TFIDFSearchInput, TFIDFSearchOptions};
use crate::indexes::{IndexOps, SearchResult};
use crate::models::types::{DocumentId, VectorId};

pub(crate) async fn dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::DenseSearchRequestDto,
) -> Result<(Vec<SearchResult>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("HNSW index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        hnsw_index
            .search(
                &collection,
                DenseSearchInput(request.query_vector, request.filter),
                &DenseSearchOptions {
                    top_k: request.top_k,
                },
                &ctx.config,
                request.return_raw_text,
            )
            .map_err(SearchError::WaCustom)?,
        warning,
    ))
}

pub(crate) async fn batch_dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchDenseSearchRequestDto,
) -> Result<(Vec<Vec<SearchResult>>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("HNSW index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        hnsw_index
            .batch_search(
                &collection,
                request
                    .queries
                    .into_iter()
                    .map(|query| DenseSearchInput(query.vector, query.filter))
                    .collect(),
                &DenseSearchOptions {
                    top_k: request.top_k,
                },
                &ctx.config,
                request.return_raw_text,
            )
            .map_err(SearchError::WaCustom)?,
        warning,
    ))
}

pub(crate) async fn sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::SparseSearchRequestDto,
) -> Result<(Vec<SearchResult>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let inverted_index = collection.get_inverted_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("Sparse index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        inverted_index
            .search(
                &collection,
                SparseSearchInput(request.query_terms),
                &SparseSearchOptions {
                    top_k: request.top_k,
                    early_terminate_threshold: request.early_terminate_threshold,
                },
                &ctx.config,
                request.return_raw_text,
            )
            .map_err(SearchError::WaCustom)?,
        warning,
    ))
}

pub(crate) async fn batch_sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchSparseSearchRequestDto,
) -> Result<(Vec<Vec<SearchResult>>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let inverted_index = collection.get_inverted_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("Sparse index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        inverted_index
            .batch_search(
                &collection,
                request
                    .query_terms_list
                    .into_iter()
                    .map(SparseSearchInput)
                    .collect(),
                &SparseSearchOptions {
                    top_k: request.top_k,
                    early_terminate_threshold: request.early_terminate_threshold,
                },
                &ctx.config,
                request.return_raw_text,
            )
            .map_err(SearchError::WaCustom)?,
        warning,
    ))
}

pub(crate) async fn hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::HybridSearchRequestDto,
) -> Result<(Vec<SearchResult>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let results_pair = match request.query {
        dtos::HybridSearchQuery::DenseAndSparse {
            query_vector,
            query_terms,
            sparse_early_terminate_threshold,
        } => {
            let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
                SearchError::IndexNotFound(format!("HNSW index for collection '{}'", collection_id))
            })?;
            let inverted_index = collection.get_inverted_index().ok_or_else(|| {
                SearchError::IndexNotFound(format!(
                    "Sparse index for collection '{}'",
                    collection_id
                ))
            })?;

            let dense_results = hnsw_index
                .search(
                    &collection,
                    DenseSearchInput(query_vector, None),
                    &DenseSearchOptions {
                        top_k: Some(request.top_k * 3),
                    },
                    &ctx.config,
                    request.return_raw_text,
                )
                .map_err(SearchError::WaCustom)?;
            let sparse_results = inverted_index
                .search(
                    &collection,
                    SparseSearchInput(query_terms),
                    &SparseSearchOptions {
                        top_k: Some(request.top_k * 3),
                        early_terminate_threshold: sparse_early_terminate_threshold,
                    },
                    &ctx.config,
                    request.return_raw_text,
                )
                .map_err(SearchError::WaCustom)?;

            (dense_results, sparse_results)
        }
        dtos::HybridSearchQuery::DenseAndTFIDF {
            query_vector,
            query_text,
        } => {
            let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
                SearchError::IndexNotFound(format!("HNSW index for collection '{}'", collection_id))
            })?;
            let tf_idf_index = collection.get_tf_idf_index().ok_or_else(|| {
                SearchError::IndexNotFound(format!(
                    "TF-IDF index for collection '{}'",
                    collection_id
                ))
            })?;

            let dense_results = hnsw_index
                .search(
                    &collection,
                    DenseSearchInput(query_vector, None),
                    &DenseSearchOptions {
                        top_k: Some(request.top_k * 3),
                    },
                    &ctx.config,
                    request.return_raw_text,
                )
                .map_err(SearchError::WaCustom)?;
            let tf_idf_results = tf_idf_index
                .search(
                    &collection,
                    TFIDFSearchInput(query_text),
                    &TFIDFSearchOptions {
                        top_k: Some(request.top_k * 3),
                    },
                    &ctx.config,
                    request.return_raw_text,
                )
                .map_err(SearchError::WaCustom)?;

            (dense_results, tf_idf_results)
        }
        dtos::HybridSearchQuery::SparseAndTFIDF {
            query_terms,
            query_text,
            sparse_early_terminate_threshold,
        } => {
            let inverted_index = collection.get_inverted_index().ok_or_else(|| {
                SearchError::IndexNotFound(format!(
                    "Sparse index for collection '{}'",
                    collection_id
                ))
            })?;
            let tf_idf_index = collection.get_tf_idf_index().ok_or_else(|| {
                SearchError::IndexNotFound(format!(
                    "TF-IDF index for collection '{}'",
                    collection_id
                ))
            })?;

            let sparse_results = inverted_index
                .search(
                    &collection,
                    SparseSearchInput(query_terms),
                    &SparseSearchOptions {
                        top_k: Some(request.top_k * 3),
                        early_terminate_threshold: sparse_early_terminate_threshold,
                    },
                    &ctx.config,
                    request.return_raw_text,
                )
                .map_err(SearchError::WaCustom)?;
            let tf_idf_results = tf_idf_index
                .search(
                    &collection,
                    TFIDFSearchInput(query_text),
                    &TFIDFSearchOptions {
                        top_k: Some(request.top_k * 3),
                    },
                    &ctx.config,
                    request.return_raw_text,
                )
                .map_err(SearchError::WaCustom)?;

            (sparse_results, tf_idf_results)
        }
    };

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    let mut final_scores: FxHashMap<VectorId, (f32, Option<DocumentId>, Option<String>)> =
        FxHashMap::default();
    let k = request.fusion_constant_k;
    if k < 0.0 {
        log::warn!("RRF fusion_constant_k ({}) is non-positive.", k);
    }

    for (rank, (vector_id, document_id, _score, text)) in results_pair.0.into_iter().enumerate() {
        let score = 1.0 / (rank as f32 + k + f32::EPSILON);
        final_scores.insert(vector_id, (score, document_id, text));
    }

    for (rank, (vector_id, document_id, _score, text)) in results_pair.1.into_iter().enumerate() {
        let score = 1.0 / (rank as f32 + k + f32::EPSILON);
        final_scores
            .entry(vector_id)
            .or_insert((0.0, document_id, text))
            .0 += score;
    }

    let mut final_results: Vec<(VectorId, Option<DocumentId>, f32, Option<String>)> = final_scores
        .into_iter()
        .map(|(id, (score, document_id, text))| (id, document_id, score, text))
        .collect();

    if final_results.len() > request.top_k {
        final_results.select_nth_unstable_by(request.top_k, |a, b| b.2.total_cmp(&a.2));
    }
    final_results.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));
    Ok((final_results, warning))
}

pub(crate) async fn batch_hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchHybridSearchRequestDto,
) -> Result<(Vec<Vec<SearchResult>>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    // Separate queries by type for batch processing
    let queries_count = request.queries.len();
    let mut dense_queries = Vec::new();
    let mut sparse_queries = Vec::new();
    let mut tfidf_queries = Vec::new();
    let mut query_mapping = Vec::new(); // Track which queries use which types

    for (query_idx, query) in request.queries.into_iter().enumerate() {
        match query {
            dtos::HybridSearchQuery::DenseAndSparse {
                query_vector,
                query_terms,
                sparse_early_terminate_threshold,
            } => {
                query_mapping.push((
                    query_idx,
                    dense_queries.len(),
                    sparse_queries.len(),
                    "dense_sparse",
                    sparse_early_terminate_threshold,
                ));
                dense_queries.push(dtos::BatchDenseSearchRequestQueryDto {
                    vector: query_vector,
                    filter: None,
                });
                sparse_queries.push(query_terms);
            }
            dtos::HybridSearchQuery::DenseAndTFIDF {
                query_vector,
                query_text,
            } => {
                query_mapping.push((
                    query_idx,
                    dense_queries.len(),
                    tfidf_queries.len(),
                    "dense_tfidf",
                    None,
                ));
                dense_queries.push(dtos::BatchDenseSearchRequestQueryDto {
                    vector: query_vector,
                    filter: None,
                });
                tfidf_queries.push(query_text);
            }
            dtos::HybridSearchQuery::SparseAndTFIDF {
                query_terms,
                query_text,
                sparse_early_terminate_threshold,
            } => {
                query_mapping.push((
                    query_idx,
                    sparse_queries.len(),
                    tfidf_queries.len(),
                    "sparse_tfidf",
                    sparse_early_terminate_threshold,
                ));
                sparse_queries.push(query_terms);
                tfidf_queries.push(query_text);
            }
        }
    }

    // Call batch functions in parallel using tokio::join
    let (dense_results, sparse_results, tfidf_results) = tokio::try_join!(
        async {
            if !dense_queries.is_empty() {
                batch_dense_search(
                    ctx.clone(),
                    collection_id,
                    dtos::BatchDenseSearchRequestDto {
                        queries: dense_queries,
                        top_k: Some(request.top_k * 3), // Same as hybrid_search
                        return_raw_text: request.return_raw_text,
                    },
                )
                .await
            } else {
                Ok((vec![], None))
            }
        },
        async {
            if !sparse_queries.is_empty() {
                batch_sparse_search(
                    ctx.clone(),
                    collection_id,
                    dtos::BatchSparseSearchRequestDto {
                        query_terms_list: sparse_queries,
                        top_k: Some(request.top_k * 3),
                        early_terminate_threshold: None,
                        return_raw_text: request.return_raw_text,
                    },
                )
                .await
            } else {
                Ok((vec![], None))
            }
        },
        async {
            if !tfidf_queries.is_empty() {
                batch_tf_idf_search(
                    ctx.clone(),
                    collection_id,
                    dtos::BatchSearchTFIDFDocumentsDto {
                        queries: tfidf_queries,
                        top_k: Some(request.top_k * 3), // Same as hybrid_search
                        return_raw_text: request.return_raw_text,
                    },
                )
                .await
            } else {
                Ok((vec![], None))
            }
        }
    )?;

    // Apply fusion logic to combine results
    let mut final_results = vec![Vec::new(); queries_count];

    for (query_idx, idx1, idx2, query_type, _sparse_threshold) in query_mapping {
        let mut final_scores: FxHashMap<VectorId, (f32, Option<DocumentId>, Option<String>)> =
            FxHashMap::default();
        let k = request.fusion_constant_k;

        if k < 0.0 {
            log::warn!("RRF fusion_constant_k ({}) is non-positive.", k);
        }

        // Get results for this query based on type
        let (first_results, second_results) = match query_type {
            "dense_sparse" => {
                let dense_idx = idx1;
                let sparse_idx = idx2;
                (
                    dense_results.0.get(dense_idx).cloned().unwrap_or_default(),
                    sparse_results
                        .0
                        .get(sparse_idx)
                        .cloned()
                        .unwrap_or_default(),
                )
            }
            "dense_tfidf" => {
                let dense_idx = idx1;
                let tfidf_idx = idx2;
                (
                    dense_results.0.get(dense_idx).cloned().unwrap_or_default(),
                    tfidf_results.0.get(tfidf_idx).cloned().unwrap_or_default(),
                )
            }
            "sparse_tfidf" => {
                let sparse_idx = idx1;
                let tfidf_idx = idx2;
                (
                    sparse_results
                        .0
                        .get(sparse_idx)
                        .cloned()
                        .unwrap_or_default(),
                    tfidf_results.0.get(tfidf_idx).cloned().unwrap_or_default(),
                )
            }
            _ => unreachable!(),
        };

        // Apply RRF fusion (same logic as hybrid_search)
        for (rank, (vector_id, document_id, _score, text)) in first_results.into_iter().enumerate()
        {
            let score = 1.0 / (rank as f32 + k + f32::EPSILON);
            final_scores.insert(vector_id, (score, document_id, text));
        }

        for (rank, (vector_id, document_id, _score, text)) in second_results.into_iter().enumerate()
        {
            let score = 1.0 / (rank as f32 + k + f32::EPSILON);
            final_scores
                .entry(vector_id)
                .and_modify(|e| e.0 += score)
                .or_insert((score, document_id, text));
        }

        // Convert to final results
        let mut query_results: Vec<(VectorId, Option<DocumentId>, f32, Option<String>)> =
            final_scores
                .into_iter()
                .map(|(id, (score, document_id, text))| (id, document_id, score, text))
                .collect();

        if query_results.len() > request.top_k {
            query_results.select_nth_unstable_by(request.top_k, |a, b| b.2.total_cmp(&a.2));
        }
        query_results.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));

        final_results[query_idx] = query_results;
    }

    Ok((final_results, warning))
}

pub(crate) async fn tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::FindSimilarTFIDFDocumentDto,
) -> Result<(Vec<SearchResult>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let tf_idf_index = collection.get_tf_idf_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("TF-IDF index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        tf_idf_index
            .search(
                &collection,
                TFIDFSearchInput(request.query),
                &TFIDFSearchOptions {
                    top_k: request.top_k,
                },
                &ctx.config,
                request.return_raw_text,
            )
            .map_err(SearchError::WaCustom)?,
        warning,
    ))
}

pub(crate) async fn batch_tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchSearchTFIDFDocumentsDto,
) -> Result<(Vec<Vec<SearchResult>>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let tf_idf_index = collection.get_tf_idf_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("TF-IDF index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        tf_idf_index
            .batch_search(
                &collection,
                request.queries.into_iter().map(TFIDFSearchInput).collect(),
                &TFIDFSearchOptions {
                    top_k: request.top_k,
                },
                &ctx.config,
                request.return_raw_text,
            )
            .map_err(SearchError::WaCustom)?,
        warning,
    ))
}
