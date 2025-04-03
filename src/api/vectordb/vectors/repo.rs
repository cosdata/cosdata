use std::sync::{atomic::Ordering, Arc};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    api_service::{
        ann_vector_query, batch_ann_vector_query, run_upload_sparse_idf_vectors,
        run_upload_sparse_idf_vectors_in_transaction, run_upload_sparse_vectors,
        run_upload_sparse_vectors_in_transaction,
    },
    distance::dotproduct::DotProductDistance,
    indexes::{
        hnsw::{transaction::HNSWIndexTransaction, HNSWIndex},
        inverted::{transaction::InvertedIndexTransaction, types::SparsePair, InvertedIndex},
        inverted_idf::{transaction::InvertedIndexIDFTransaction, InvertedIndexIDF},
    },
    models::{
        common::WaCustomError,
        rpc::DenseVector,
        sparse_ann_query::{SparseAnnIDFResult, SparseAnnQueryBasic, SparseAnnResult},
        types::{MetricResult, SparseVector, VectorId},
    },
    vector_store::get_sparse_embedding_by_id,
};

use crate::{
    api::vectordb::collections,
    api_service::{run_upload_dense_vectors, run_upload_dense_vectors_in_transaction},
    app_context::AppContext,
    vector_store::get_dense_embedding_by_id,
};

use super::{
    dtos::{
        BatchSearchDenseVectorsDto, BatchSearchSparseVectorsDto, CreateDenseVectorDto,
        CreateSparseVectorDto, FindSimilarDenseVectorsDto, FindSimilarSparseVectorsDto,
        FindSimilarVectorsResponseDto, SimilarVector, UpdateVectorDto, UpdateVectorResponseDto,
    },
    error::VectorsError,
};

/// Creates a sparse vector for inverted index
///
pub(crate) async fn create_sparse_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateSparseVectorDto,
) -> Result<(), VectorsError> {
    if let Some(inverted_index) = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
    {
        if !inverted_index
            .current_open_transaction
            .load(Ordering::SeqCst)
            .is_null()
        {
            return Err(VectorsError::FailedToCreateVector(
                "there is an ongoing transaction!".into(),
            ));
        }

        run_upload_sparse_vectors(
            inverted_index,
            vec![(
                create_vector_dto.id.clone(),
                create_vector_dto.values.clone(),
            )],
        )
        .map_err(VectorsError::WaCustom)?;
    } else {
        let inverted_index = ctx
            .ain_env
            .collections_map
            .get_idf_inverted_index(collection_id)
            .ok_or(VectorsError::IndexNotFound)?;
        if !inverted_index
            .current_open_transaction
            .load(Ordering::SeqCst)
            .is_null()
        {
            return Err(VectorsError::FailedToCreateVector(
                "there is an ongoing transaction!".into(),
            ));
        }

        run_upload_sparse_idf_vectors(
            inverted_index,
            vec![(
                create_vector_dto.id.clone(),
                create_vector_dto.values.clone(),
            )],
        )
        .map_err(VectorsError::WaCustom)?;
    }

    Ok(())
}

/// Creates a vector for dense index
///
pub(crate) async fn create_dense_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateDenseVectorDto,
) -> Result<(), VectorsError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(VectorsError::IndexNotFound)?;

    if !hnsw_index
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return Err(VectorsError::FailedToCreateVector(
            "there is an ongoing transaction!".into(),
        ));
    }

    run_upload_dense_vectors(
        ctx,
        hnsw_index,
        vec![(create_vector_dto.id, create_vector_dto.values)],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(())
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction: &HNSWIndexTransaction,
    create_vector_dto: CreateDenseVectorDto,
) -> Result<(), VectorsError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(VectorsError::IndexNotFound)?;

    run_upload_dense_vectors_in_transaction(
        ctx.clone(),
        hnsw_index,
        transaction,
        vec![(create_vector_dto.id, create_vector_dto.values)],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(())
}

pub(crate) async fn get_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<CreateDenseVectorDto, VectorsError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(VectorsError::IndexNotFound)?;

    let embedding = get_dense_embedding_by_id(hnsw_index, &vector_id)
        .map_err(|e| VectorsError::DatabaseError(e.to_string()))?;

    let id = embedding.hash_vec;

    Ok(CreateDenseVectorDto {
        id,
        values: (*embedding.raw_vec).clone(),
    })
}

pub(crate) async fn update_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
    update_vector_dto: UpdateVectorDto,
) -> Result<UpdateVectorResponseDto, VectorsError> {
    let hnsw_index = collections::service::get_dense_index_by_id(ctx.clone(), collection_id)
        .await
        .map_err(|e| VectorsError::FailedToUpdateVector(e.to_string()))?;

    if !hnsw_index
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return Err(VectorsError::FailedToUpdateVector(
            "there is an ongoing transaction!".into(),
        ));
    }

    run_upload_dense_vectors(
        ctx,
        hnsw_index,
        vec![(vector_id.clone(), update_vector_dto.values.clone())],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(UpdateVectorResponseDto {
        id: vector_id,
        values: update_vector_dto.values,
    })
}

pub(crate) async fn find_similar_dense_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    find_similar_vectors: FindSimilarDenseVectorsDto,
) -> Result<FindSimilarVectorsResponseDto, VectorsError> {
    if find_similar_vectors.vector.is_empty() {
        return Err(VectorsError::FailedToFindSimilarVectors(
            "Vector shouldn't be empty".to_string(),
        ));
    }

    let hnsw_index = match ctx.ain_env.collections_map.get_hnsw_index(collection_id) {
        Some(store) => store,
        None => {
            return Err(VectorsError::IndexNotFound);
        }
    };

    let result = match ann_vector_query(
        ctx,
        hnsw_index,
        find_similar_vectors.vector,
        find_similar_vectors.k,
    )
    .await
    {
        Ok(results) => results,
        Err(err) => return Err(VectorsError::FailedToFindSimilarVectors(err.to_string())),
    };
    Ok(FindSimilarVectorsResponseDto {
        results: result
            .into_iter()
            .map(|(id, sim)| SimilarVector {
                id,
                score: sim.get_value(),
            })
            .collect(),
    })
}

pub(crate) async fn find_similar_sparse_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    find_similar_vectors: FindSimilarSparseVectorsDto,
) -> Result<FindSimilarVectorsResponseDto, VectorsError> {
    if find_similar_vectors.values.is_empty() {
        return Err(VectorsError::FailedToFindSimilarVectors(
            "Vector shouldn't be empty".to_string(),
        ));
    }

    let results = if let Some(inverted_index) = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
    {
        sparse_ann_vector_query(
            ctx,
            inverted_index,
            &find_similar_vectors.values,
            find_similar_vectors.top_k,
        )
        .map_err(|e| VectorsError::FailedToFindSimilarVectors(e.to_string()))?
    } else {
        let inverted_index_idf = ctx
            .ain_env
            .collections_map
            .get_idf_inverted_index(collection_id)
            .ok_or(VectorsError::IndexNotFound)?;

        sparse_idf_ann_vector_query(
            ctx,
            inverted_index_idf,
            &find_similar_vectors.values,
            find_similar_vectors.top_k,
        )
        .map_err(|e| VectorsError::FailedToFindSimilarVectors(e.to_string()))?
    };

    Ok(FindSimilarVectorsResponseDto {
        results: results
            .into_iter()
            .map(|(id, sim)| SimilarVector {
                id,
                score: sim.get_value(),
            })
            .collect(),
    })
}

pub(crate) async fn batch_search_dense_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    batch_search_vectors: BatchSearchDenseVectorsDto,
) -> Result<Vec<FindSimilarVectorsResponseDto>, VectorsError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(VectorsError::IndexNotFound)?;

    let results = batch_ann_vector_query(
        ctx,
        hnsw_index,
        batch_search_vectors.vectors,
        batch_search_vectors.k,
    )
    .await
    .map_err(|err| VectorsError::FailedToFindSimilarVectors(err.to_string()))?;

    Ok(results
        .into_iter()
        .map(|results| FindSimilarVectorsResponseDto {
            results: results
                .into_iter()
                .map(|(id, sim)| SimilarVector {
                    id,
                    score: sim.get_value(),
                })
                .collect(),
        })
        .collect())
}

pub(crate) async fn batch_search_sparse_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    batch_search_vectors: BatchSearchSparseVectorsDto,
) -> Result<Vec<FindSimilarVectorsResponseDto>, VectorsError> {
    let results = if let Some(inverted_index) = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
    {
        batch_sparse_ann_vector_query(
            ctx,
            inverted_index,
            &batch_search_vectors.vectors,
            batch_search_vectors.top_k,
        )
        .map_err(|e| VectorsError::FailedToFindSimilarVectors(e.to_string()))?
    } else {
        let inverted_index_idf = ctx
            .ain_env
            .collections_map
            .get_idf_inverted_index(collection_id)
            .ok_or(VectorsError::IndexNotFound)?;

        batch_sparse_idf_ann_vector_query(
            ctx,
            inverted_index_idf,
            &batch_search_vectors.vectors,
            batch_search_vectors.top_k,
        )
        .map_err(|e| VectorsError::FailedToFindSimilarVectors(e.to_string()))?
    };

    Ok(results
        .into_iter()
        .map(|result| FindSimilarVectorsResponseDto {
            results: result
                .into_iter()
                .map(|(id, sim)| SimilarVector {
                    id,
                    score: sim.get_value(),
                })
                .collect(),
        })
        .collect())
}

fn sparse_ann_vector_query(
    ctx: Arc<AppContext>,
    inverted_index: Arc<InvertedIndex>,
    query: &[SparsePair],
    top_k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    // create a query to get similar sparse vectors
    let sparse_vec = SparseVector {
        vector_id: u32::MAX,
        entries: query.iter().map(|pair| (pair.0, pair.1)).collect(),
    };

    let intermediate_results = SparseAnnQueryBasic::new(sparse_vec).sequential_search(
        &inverted_index.root,
        inverted_index.root.root.quantization_bits,
        *inverted_index.values_upper_bound.read().unwrap(),
        inverted_index.early_terminate_threshold,
        if ctx.config.rerank_sparse_with_raw_values {
            ctx.config.sparse_raw_values_reranking_factor
        } else {
            1
        },
        top_k,
    )?;

    if ctx.config.rerank_sparse_with_raw_values {
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

fn sparse_idf_ann_vector_query(
    ctx: Arc<AppContext>,
    inverted_index_idf: Arc<InvertedIndexIDF>,
    query: &[SparsePair],
    top_k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    // create a query to get similar sparse vectors
    let sparse_vec = SparseVector {
        vector_id: u32::MAX,
        entries: query.iter().map(|pair| (pair.0, pair.1)).collect(),
    };

    let (intermediate_results, idf_list) = SparseAnnQueryBasic::new(sparse_vec).search_bm25(
        &inverted_index_idf.root,
        inverted_index_idf.root.root.quantization_bits,
        inverted_index_idf.early_terminate_threshold,
        if ctx.config.rerank_sparse_with_raw_values {
            ctx.config.sparse_raw_values_reranking_factor
        } else {
            1
        },
        top_k,
    )?;

    if ctx.config.rerank_sparse_with_raw_values {
        finalize_sparse_idf_ann_results(
            inverted_index_idf,
            intermediate_results,
            idf_list,
            query,
            top_k,
        )
    } else {
        Ok(intermediate_results
            .into_iter()
            .map(|result| {
                (
                    VectorId(result.document_id as u64),
                    MetricResult::DotProductDistance(DotProductDistance(result.score)),
                )
            })
            .collect())
    }
}

fn batch_sparse_ann_vector_query(
    ctx: Arc<AppContext>,
    inverted_index: Arc<InvertedIndex>,
    queries: &[Vec<SparsePair>],
    top_k: Option<usize>,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    queries
        .par_iter()
        .map(|query| sparse_ann_vector_query(ctx.clone(), inverted_index.clone(), query, top_k))
        .collect()
}

fn batch_sparse_idf_ann_vector_query(
    ctx: Arc<AppContext>,
    inverted_index_idf: Arc<InvertedIndexIDF>,
    queries: &[Vec<SparsePair>],
    top_k: Option<usize>,
) -> Result<Vec<Vec<(VectorId, MetricResult)>>, WaCustomError> {
    queries
        .par_iter()
        .map(|query| {
            sparse_idf_ann_vector_query(ctx.clone(), inverted_index_idf.clone(), query, top_k)
        })
        .collect()
}

fn finalize_sparse_ann_results(
    inverted_index: Arc<InvertedIndex>,
    intermediate_results: Vec<SparseAnnResult>,
    query: &[SparsePair],
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let mut results = Vec::with_capacity(k.unwrap_or(intermediate_results.len()));

    for result in intermediate_results {
        let id = VectorId(result.vector_id as u64);
        let map =
            get_sparse_embedding_by_id(&inverted_index.lmdb, &inverted_index.vec_raw_manager, &id)?
                .into_map();
        let mut dp = 0.0;
        for pair in query {
            if let Some(val) = map.get(&pair.0) {
                dp += val * pair.1;
            }
        }
        results.push((id, MetricResult::DotProductDistance(DotProductDistance(dp))));
    }

    results.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
    if let Some(k) = k {
        results.truncate(k);
    }

    Ok(results)
}

fn finalize_sparse_idf_ann_results(
    inverted_index_idf: Arc<InvertedIndexIDF>,
    intermediate_results: Vec<SparseAnnIDFResult>,
    idf_list: Vec<f32>,
    query: &[SparsePair],
    k: Option<usize>,
) -> Result<Vec<(VectorId, MetricResult)>, WaCustomError> {
    let mut results = Vec::with_capacity(k.unwrap_or(intermediate_results.len()));

    for result in intermediate_results {
        let id = VectorId(result.document_id as u64);
        let map = get_sparse_embedding_by_id(
            &inverted_index_idf.lmdb,
            &inverted_index_idf.vec_raw_manager,
            &id,
        )?
        .into_map();
        let mut score = 0.0;
        for (idx, pair) in query.iter().enumerate() {
            if let Some(tf) = map.get(&pair.0) {
                score += tf * idf_list[idx];
            }
        }
        results.push((
            id,
            MetricResult::DotProductDistance(DotProductDistance(score)),
        ));
    }

    results.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
    if let Some(k) = k {
        results.truncate(k);
    }

    Ok(results)
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    _vector_id: u64,
) -> Result<(), VectorsError> {
    let _collection = collections::service::get_dense_index_by_id(ctx.clone(), collection_id)
        .await
        .map_err(|e| VectorsError::FailedToDeleteVector(e.to_string()))?;

    unimplemented!();
}

pub(crate) async fn upsert_dense_vectors_in_transaction(
    ctx: Arc<AppContext>,
    hnsw_index: Arc<HNSWIndex>,
    transaction: &HNSWIndexTransaction,
    vectors: Vec<DenseVector>,
) -> Result<(), VectorsError> {
    run_upload_dense_vectors_in_transaction(
        ctx.clone(),
        hnsw_index,
        transaction,
        vectors
            .into_iter()
            .map(|vec| (vec.id, vec.values))
            .collect(),
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(())
}

pub(crate) async fn upsert_sparse_vectors_in_transaction(
    ctx: Arc<AppContext>,
    inverted_index: Arc<InvertedIndex>,
    transaction: &InvertedIndexTransaction,
    vectors: Vec<CreateSparseVectorDto>,
) -> Result<(), VectorsError> {
    run_upload_sparse_vectors_in_transaction(
        ctx,
        inverted_index,
        transaction,
        vectors
            .into_iter()
            .map(|vec| (vec.id, vec.values))
            .collect(),
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(())
}

pub(crate) async fn upsert_sparse_idf_vectors_in_transaction(
    inverted_index: Arc<InvertedIndexIDF>,
    transaction: &InvertedIndexIDFTransaction,
    vectors: Vec<CreateSparseVectorDto>,
) -> Result<(), VectorsError> {
    run_upload_sparse_idf_vectors_in_transaction(
        inverted_index,
        transaction,
        vectors
            .into_iter()
            .map(|vec| (vec.id, vec.values))
            .collect(),
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(())
}
