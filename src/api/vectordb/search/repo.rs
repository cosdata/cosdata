use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::dtos;
use super::error::SearchError;
use crate::app_context::AppContext;
use crate::indexes::hnsw::{DenseSearchInput, DenseSearchOptions};
use crate::indexes::inverted::{SparseSearchInput, SparseSearchOptions};
use crate::indexes::tf_idf::{TFIDFSearchInput, TFIDFSearchOptions};
use crate::indexes::{IndexOps, Matches, SearchResult};
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

pub(crate) async fn geofence_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::GeoFenceSearchRequestDto,
) -> Result<(Vec<SearchResult>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let inverted_index = collection.get_inverted_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("inverted index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        inverted_index
            .search(
                &collection,
                SparseSearchInput(request.query),
                &SparseSearchOptions {
                    sort_by_distance: request.sort_by_distance,
                    coordinates: request.coordinates,
                    zones: request.zones,
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

pub(crate) async fn batch_geofence_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchGeoFenceSearchRequestDto,
) -> Result<(Vec<Vec<SearchResult>>, Option<String>), SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| SearchError::CollectionNotFound(collection_id.to_string()))?;

    let inverted_index = collection.get_inverted_index().ok_or_else(|| {
        SearchError::IndexNotFound(format!("sparse index for collection '{}'", collection_id))
    })?;

    let warning = collection.is_indexing().then(|| {
        "Embeddings are currently being indexed; some results may be temporarily unavailable."
            .to_string()
    });

    Ok((
        inverted_index
            .batch_search(
                &collection,
                request.queries.into_iter().map(SparseSearchInput).collect(),
                &SparseSearchOptions {
                    sort_by_distance: request.sort_by_distance,
                    coordinates: request.coordinates,
                    zones: request.zones,
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
            let sparse_results = todo!();

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

            let sparse_results = todo!();
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

    let mut final_scores: FxHashMap<
        VectorId,
        (f32, Option<DocumentId>, Option<String>, Option<Matches>),
    > = FxHashMap::default();
    let k = request.fusion_constant_k;
    if k < 0.0 {
        log::warn!("RRF fusion_constant_k ({}) is non-positive.", k);
    }

    for (rank, (vector_id, document_id, _score, text, matches)) in
        results_pair.0.into_iter().enumerate()
    {
        let score = 1.0 / (rank as f32 + k + f32::EPSILON);
        final_scores.insert(vector_id, (score, document_id, text, matches));
    }

    for (rank, (vector_id, document_id, _score, text, matches)) in
        results_pair.1.into_iter().enumerate()
    {
        let score = 1.0 / (rank as f32 + k + f32::EPSILON);
        final_scores
            .entry(vector_id)
            .or_insert((0.0, document_id, text, matches))
            .0 += score;
    }

    let mut final_results: Vec<(
        VectorId,
        Option<DocumentId>,
        f32,
        Option<String>,
        Option<Matches>,
    )> = final_scores
        .into_iter()
        .map(|(id, (score, document_id, text, matches))| (id, document_id, score, text, matches))
        .collect();

    if final_results.len() > request.top_k {
        final_results.select_nth_unstable_by(request.top_k, |a, b| b.2.total_cmp(&a.2));
    }
    final_results.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));
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
