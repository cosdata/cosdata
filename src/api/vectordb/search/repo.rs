use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::sync::Arc;

use super::dtos;
use super::error::SearchError;
use crate::indexes::tf_idf::TFIDFIndex;
use crate::metadata::query_filtering::Filter;
use crate::{
    api_service::{ann_vector_query, batch_ann_vector_query},
    app_context::AppContext,
    config_loader::Config,
    distance::dotproduct::DotProductDistance,
    indexes::{inverted::types::SparsePair, inverted::InvertedIndex, tf_idf::process_text},
    models::{
        common::WaCustomError,
        sparse_ann_query::{SparseAnnQueryBasic, SparseAnnResult},
        types::{MetricResult, SparseVector, VectorId},
    },
};

#[allow(dead_code)]
pub(crate) async fn dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::DenseSearchRequestDto,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("collection '{}'", collection_id)))?;

    let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
        WaCustomError::NotFound(format!(
            "Dense index not found for collection '{}'",
            collection_id
        ))
    })?;

    let metadata_filter: Option<Filter> = request.filter;

    ann_vector_query(
        ctx,
        &collection,
        hnsw_index.clone(),
        request.query_vector,
        metadata_filter,
        request.top_k,
    )
    .await
}

#[allow(dead_code)]
pub(crate) async fn batch_dense_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchDenseSearchRequestDto,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("collection '{}'", collection_id)))?;

    let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
        WaCustomError::NotFound(format!(
            "Dense index not found for collection '{}'",
            collection_id
        ))
    })?;

    let metadata_filter: Option<Filter> = request.filter;

    batch_ann_vector_query(
        ctx,
        &collection,
        hnsw_index.clone(),
        request.query_vectors,
        metadata_filter,
        request.top_k,
    )
    .await
}

pub(crate) async fn sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::SparseSearchRequestDto,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("collection '{}'", collection_id)))?;

    let inverted_index = collection.get_inverted_index().ok_or_else(|| {
        WaCustomError::NotFound(format!(
            "Sparse index not found for collection '{}'",
            collection_id
        ))
    })?;

    let threshold = request
        .early_terminate_threshold
        .unwrap_or(ctx.config.search.early_terminate_threshold);
    // Directly call the logic for regular sparse
    sparse_ann_vector_query_logic(
        &ctx.config,
        inverted_index.clone(),
        &request.query_terms,
        request.top_k,
        threshold,
    )
}

pub(crate) async fn batch_sparse_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchSparseSearchRequestDto,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("collection '{}'", collection_id)))?;

    let inverted_index = collection.get_inverted_index().ok_or_else(|| {
        WaCustomError::NotFound(format!(
            "Sparse index not found for collection '{}'",
            collection_id
        ))
    })?;

    let threshold = request
        .early_terminate_threshold
        .unwrap_or(ctx.config.search.early_terminate_threshold);
    // Directly call the logic for regular sparse batch
    batch_sparse_ann_vector_query_logic(
        &ctx.config,
        inverted_index.clone(),
        &request.query_terms_list,
        request.top_k,
        threshold,
    )
}

pub fn sparse_ann_vector_query_logic(
    config: &Config,
    inverted_index: Arc<InvertedIndex>,
    query: &[SparsePair],
    top_k: Option<usize>,
    early_terminate_threshold: f32,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let sparse_vec = SparseVector {
        vector_id: u32::MAX,
        entries: query.iter().map(|pair| (pair.0, pair.1)).collect(),
    };

    let intermediate_results = SparseAnnQueryBasic::new(sparse_vec).sequential_search(
        &inverted_index.root,
        inverted_index.root.root.quantization_bits,
        *inverted_index.values_upper_bound.read().unwrap(),
        early_terminate_threshold,
        if config.rerank_sparse_with_raw_values {
            config.sparse_raw_values_reranking_factor
        } else {
            1
        },
        top_k,
    )?;

    if config.rerank_sparse_with_raw_values {
        finalize_sparse_ann_results(inverted_index, intermediate_results, query, top_k)
    } else {
        Ok(intermediate_results
            .into_iter()
            .map(|result| {
                (
                    VectorId(result.vector_id as u64),
                    MetricResult::DotProductDistance(DotProductDistance(result.similarity as f32)),
                )
            })
            .collect())
    }
}

// Synchronous batch helper
fn batch_sparse_ann_vector_query_logic(
    config: &Config,
    inverted_index: Arc<InvertedIndex>,
    queries: &[Vec<SparsePair>],
    top_k: Option<usize>,
    early_terminate_threshold: f32,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    queries
        .par_iter()
        .map(|query| {
            sparse_ann_vector_query_logic(
                config,
                inverted_index.clone(),
                query,
                top_k,
                early_terminate_threshold,
            )
        })
        .collect()
}

// Change return type back to MetricResult
fn finalize_sparse_ann_results(
    inverted_index: Arc<InvertedIndex>,
    intermediate_results: Vec<SparseAnnResult>,
    query: &[SparsePair],
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    // <-- Return MetricResult
    let mut results = Vec::with_capacity(k.unwrap_or(intermediate_results.len()));

    for result in intermediate_results {
        let vector_u64_id = result.vector_id as u64;
        let vector_id_obj = VectorId(vector_u64_id);

        match inverted_index.vec_raw_map.get_latest(vector_u64_id) {
            Some(raw_sparse_embedding_ref) => {
                let sparse_pairs = raw_sparse_embedding_ref.raw_vec.clone();
                let map: std::collections::HashMap<u32, f32> =
                    sparse_pairs.iter().map(|sp| (sp.0, sp.1)).collect();

                let mut dp = 0.0;
                for pair in query {
                    if let Some(val) = map.get(&pair.0) {
                        dp += val * pair.1;
                    }
                }
                // Wrap the f32 score (dp) in MetricResult::DotProductDistance
                results.push((
                    vector_id_obj,
                    MetricResult::DotProductDistance(
                        crate::distance::dotproduct::DotProductDistance(dp),
                    ), // <-- Wrap score
                ));
            }
            None => {
                log::warn!(
                    "Raw sparse vector ID {} (u64: {}) not found in vec_raw_map during finalization.",
                    vector_id_obj,
                    vector_u64_id
                );
            }
        }
    }

    // Sort by MetricResult (descending order)
    results.sort_unstable_by(|(_, a), (_, b)| b.cmp(a)); // cmp works for MetricResult

    if let Some(k_val) = k {
        results.truncate(k_val);
    }

    Ok(results)
}

pub(crate) async fn hybrid_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::HybridSearchRequestDto,
) -> Result<Vec<(VectorId, f32)>, SearchError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("Collection '{}'", collection_id)))?;

    let hnsw_index = collection.get_hnsw_index().ok_or_else(|| {
        SearchError::IndexNotFound("Dense index required for hybrid search.".to_string())
    })?;

    // Perform Search on *Available* Sparse Index (Synchronous Call)
    let sparse_results: Vec<(VectorId, MetricResult)> = if let Some(inverted_index) =
        collection.get_inverted_index()
    {
        let sparse_k = request.top_k * 3;
        let threshold = ctx.config.search.early_terminate_threshold;
        // Call synchronous helper
        sparse_ann_vector_query_logic(
            &ctx.config,
            inverted_index.clone(),
            &request.query_terms,
            Some(sparse_k),
            threshold,
        )
        .map_err(|e| {
            SearchError::SearchFailed(format!("Hybrid: Sparse component (regular) failed: {}", e))
        })?
    } else if let Some(idf_index) = collection.get_tf_idf_index() {
        log::debug!(
            "Using IDF index for hybrid sparse component in collection '{}'",
            collection_id
        );
        let sparse_k = request.top_k * 3;
        let query_sparse_vector = SparseVector {
            vector_id: u32::MAX,
            entries: request.query_terms.iter().map(|p| (p.0, p.1)).collect(),
        };
        // Call synchronous search_bm25
        SparseAnnQueryBasic::new(query_sparse_vector)
            .search_bm25(&idf_index.root, Some(sparse_k))
            .map(|idf_results| {
                idf_results
                    .into_iter()
                    .map(|res| {
                        (
                            VectorId(res.document_id as u64),
                            MetricResult::DotProductDistance(DotProductDistance(res.score)),
                        )
                    })
                    .collect()
            })
            .map_err(|e| {
                SearchError::SearchFailed(format!("Hybrid: Sparse component (IDF) failed: {}", e))
            })?
    } else {
        return Err(SearchError::IndexNotFound(
            "Sparse index (regular or IDF) required for hybrid search.".to_string(),
        ));
    };

    let dense_k = request.top_k * 3;
    let dense_results = ann_vector_query(
        ctx.clone(),
        &collection,
        hnsw_index.clone(),
        request.query_vector,
        None, // Pass None for filter
        Some(dense_k),
    )
    .await
    .map_err(|e| SearchError::SearchFailed(format!("Hybrid: Dense component failed: {}", e)))?;

    let mut final_scores: HashMap<VectorId, f32> = HashMap::new();
    let constant_k = request.fusion_constant_k;
    if constant_k < 0.0 {
        log::warn!("RRF fusion_constant_k ({}) is non-positive.", constant_k);
    }
    for (rank, (vector_id, _score)) in dense_results.iter().enumerate() {
        let score = 1.0 / (rank as f32 + constant_k + f32::EPSILON);
        final_scores.insert(vector_id.clone(), score);
    }
    for (rank, (vector_id, _score)) in sparse_results.iter().enumerate() {
        let score = 1.0 / (rank as f32 + constant_k + f32::EPSILON);
        *final_scores.entry(vector_id.clone()).or_insert(0.0) += score;
    }

    let mut final_results: Vec<(VectorId, f32)> = final_scores.into_iter().collect();
    final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    final_results.truncate(request.top_k);

    Ok(final_results)
}

pub fn tf_idf_ann_vector_query(
    tf_idf_index: Arc<TFIDFIndex>,
    query: &str,
    top_k: Option<usize>,
) -> Result<Vec<(VectorId, f32)>, WaCustomError> {
    // Return f32 directly
    let entries = process_text(
        query,
        40, // max_token_len - consider making configurable?
        *tf_idf_index.average_document_length.read().unwrap(),
        tf_idf_index.k1,
        tf_idf_index.b,
    );
    let sparse_vec = SparseVector {
        vector_id: u32::MAX, // Placeholder ID for query
        entries,
    };

    let results = SparseAnnQueryBasic::new(sparse_vec).search_bm25(&tf_idf_index.root, top_k)?;

    // Map internal document ID back to external VectorId using vec_raw_map
    Ok(results
        .into_iter()
        .filter_map(|result| {
            // Use filter_map to handle potential misses in map
            tf_idf_index
                .vec_raw_map
                .get_latest(result.document_id as u64)
                .map(|(ext_id, _)| (ext_id.clone(), result.score)) // Get external ID
        })
        .collect())
}

fn batch_tf_idf_ann_vector_query(
    tf_idf_index: Arc<TFIDFIndex>,
    queries: &[String],
    top_k: Option<usize>,
) -> Result<Vec<Vec<(VectorId, f32)>>, WaCustomError> {
    queries
        .par_iter() // Use parallel iterator
        .map(|query| tf_idf_ann_vector_query(tf_idf_index.clone(), query, top_k))
        .collect() // Collect results
}

pub(crate) async fn tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::FindSimilarTFIDFDocumentDto,
) -> Result<Vec<(VectorId, f32)>, WaCustomError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("Collection '{}'", collection_id)))?;

    // Returns f32 score
    let tf_idf_index = collection.get_tf_idf_index().ok_or_else(|| {
        WaCustomError::NotFound(format!(
            "Sparse IDF index not found for collection '{}'",
            collection_id
        ))
    })?;

    // Call the helper directly
    tf_idf_ann_vector_query(tf_idf_index, &request.query, request.top_k)
}

pub(crate) async fn batch_tf_idf_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    request: dtos::BatchSearchTFIDFDocumentsDto,
) -> Result<Vec<Vec<(VectorId, f32)>>, WaCustomError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| WaCustomError::NotFound(format!("collection '{}'", collection_id)))?;

    // Returns f32 score
    let tf_idf_index = collection.get_tf_idf_index().ok_or_else(|| {
        WaCustomError::NotFound(format!(
            "Sparse IDF index not found for collection '{}'",
            collection_id
        ))
    })?;

    // Call the helper directly
    batch_tf_idf_ann_vector_query(tf_idf_index, &request.queries, request.top_k)
}
