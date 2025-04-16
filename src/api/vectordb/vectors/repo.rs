use std::sync::{atomic::Ordering, Arc};

use lmdb::Transaction;

use crate::{
    api_service::{
        fetch_vector_neighbors as api_fetch_neighbors, run_upload_sparse_vectors,
        run_upload_sparse_vectors_in_transaction, run_upload_tf_idf_documents,
        run_upload_tf_idf_documents_in_transaction,
    },
    indexes::{
        hnsw::{transaction::HNSWIndexTransaction, HNSWIndex},
        inverted::{transaction::InvertedIndexTransaction, InvertedIndex},
        tf_idf::{transaction::TFIDFIndexTransaction, TFIDFIndex},
    },
    models::{common::WaCustomError, rpc::DenseVector, types::VectorId},
};

use crate::{
    api::vectordb::collections,
    api_service::{run_upload_dense_vectors, run_upload_dense_vectors_in_transaction},
    app_context::AppContext,
    vector_store::get_dense_embedding_by_id,
};

use super::{
    dtos::{
        CreateDenseVectorDto, CreateSparseVectorDto, CreateTFIDFDocumentDto,
        CreateVectorResponseDto, SimilarVector, UpdateVectorDto, UpdateVectorResponseDto,
    },
    error::VectorsError,
};

// Creates a sparse vector for inverted index
pub(crate) async fn create_sparse_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateSparseVectorDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
    let inverted_index = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
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

    run_upload_sparse_vectors(
        inverted_index,
        vec![(
            create_vector_dto.id.clone(),
            create_vector_dto.values.clone(),
        )],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(CreateVectorResponseDto::Sparse(CreateSparseVectorDto {
        id: create_vector_dto.id,
        values: create_vector_dto.values,
    }))
}

// Creates a document for TF-IDF index
pub(crate) async fn create_tf_idf_document(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_document_dto: CreateTFIDFDocumentDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
    let tf_idf_index = ctx
        .ain_env
        .collections_map
        .get_tf_idf_index(collection_id)
        .ok_or(VectorsError::IndexNotFound)?;

    if !tf_idf_index
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return Err(VectorsError::FailedToCreateVector(
            "there is an ongoing transaction!".into(),
        ));
    }

    run_upload_tf_idf_documents(
        tf_idf_index,
        vec![(
            create_document_dto.id.clone(),
            create_document_dto.text.clone(),
        )],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(CreateVectorResponseDto::TfIdf(create_document_dto))
}

/// Creates a vector for dense index
///
pub(crate) async fn create_dense_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateDenseVectorDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
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
        vec![(
            create_vector_dto.id.clone(),
            create_vector_dto.values.clone(),
            create_vector_dto.metadata.clone(),
        )],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(CreateVectorResponseDto::Dense(CreateDenseVectorDto {
        id: create_vector_dto.id,
        values: create_vector_dto.values,
        metadata: create_vector_dto.metadata,
    }))
}

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection_id: &str,
    transaction: &HNSWIndexTransaction,
    create_vector_dto: CreateDenseVectorDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
    let hnsw_index = ctx
        .ain_env
        .collections_map
        .get_hnsw_index(collection_id)
        .ok_or(VectorsError::IndexNotFound)?;

    run_upload_dense_vectors_in_transaction(
        ctx.clone(),
        hnsw_index,
        transaction,
        vec![(
            create_vector_dto.id.clone(),
            create_vector_dto.values.clone(),
            create_vector_dto.metadata.clone(),
        )],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(CreateVectorResponseDto::Dense(CreateDenseVectorDto {
        id: create_vector_dto.id,
        values: create_vector_dto.values,
        metadata: create_vector_dto.metadata,
    }))
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
        metadata: embedding.raw_metadata,
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
        .map_err(|e| {
            VectorsError::FailedToUpdateVector(format!(
                "Failed to get dense index for update: {}",
                e
            ))
        })?;

    if !hnsw_index
        .current_open_transaction
        .load(Ordering::SeqCst)
        .is_null()
    {
        return Err(VectorsError::WaCustom(WaCustomError::LockError(
            "Cannot update vector while transaction is open".to_string(),
        )));
    }

    run_upload_dense_vectors(
        ctx,
        hnsw_index,
        vec![(vector_id.clone(), update_vector_dto.values.clone(), None)],
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(UpdateVectorResponseDto {
        id: vector_id,
        values: update_vector_dto.values,
    })
}

pub(crate) async fn delete_vector_by_id(
    _ctx: Arc<AppContext>,
    collection_id: &str,
    _vector_id: u64,
) -> Result<(), VectorsError> {
    log::error!(
        "Vector deletion (ID: {}) for collection '{}' is not implemented in the core index logic.",
        _vector_id,
        collection_id
    );
    Err(VectorsError::NotImplemented)
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
            // @TODO(vineet): Add support for metadata fields
            .map(|vec| (vec.id, vec.values, None))
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

pub(crate) async fn upsert_tf_idf_documents_in_transaction(
    tf_idf_index: Arc<TFIDFIndex>,
    transaction: &TFIDFIndexTransaction,
    documents: Vec<CreateTFIDFDocumentDto>,
) -> Result<(), VectorsError> {
    run_upload_tf_idf_documents_in_transaction(
        tf_idf_index,
        transaction,
        documents
            .into_iter()
            .map(|doc| (doc.id, doc.text))
            .collect(),
    )
    .map_err(VectorsError::WaCustom)?;

    Ok(())
}

pub(crate) async fn check_vector_existence(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: u64,
) -> Result<bool, VectorsError> {
    if ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .is_none()
    {
        log::debug!(
            "Collection '{}' not found during existence check for vector {}",
            collection_id,
            vector_id
        );
        return Ok(false);
    }

    let vector_id_obj = VectorId(vector_id);
    if let Some(hnsw_index) = ctx.ain_env.collections_map.get_hnsw_index(collection_id) {
        let env = hnsw_index.lmdb.env.clone();
        let db = *hnsw_index.lmdb.db;
        let txn = env.begin_ro_txn().map_err(|e| {
            VectorsError::DatabaseError(format!("LMDB RO txn failed for dense check: {}", e))
        })?;
        let key = crate::macros::key!(e: &vector_id_obj);

        match txn.get(db, &key) {
            Ok(_) => {
                txn.abort();
                return Ok(true);
            }
            Err(lmdb::Error::NotFound) => {
                txn.abort();
            }
            Err(e) => {
                txn.abort();
                log::error!(
                    "LMDB error checking dense vector existence for {}: {}",
                    vector_id,
                    e
                );
                return Err(VectorsError::DatabaseError(format!(
                    "Dense existence check failed: {}",
                    e
                )));
            }
        }
    }

    if let Some(inverted_index) = ctx
        .ain_env
        .collections_map
        .get_inverted_index(collection_id)
    {
        if inverted_index.contains_vector_id(vector_id as u32) {
            return Ok(true);
        }
    }

    if let Some(idf_index) = ctx.ain_env.collections_map.get_tf_idf_index(collection_id) {
        if idf_index.contains_vector_id(vector_id as u32) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) async fn fetch_vector_neighbors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<Vec<SimilarVector>, VectorsError> {
    let hnsw_index = collections::service::get_dense_index_by_id(ctx.clone(), collection_id)
        .await
        .map_err(|_| VectorsError::IndexNotFound)?;

    let results = api_fetch_neighbors(hnsw_index.clone(), vector_id.clone()).await;
    let neighbors: Vec<SimilarVector> = results
        .into_iter()
        .filter_map(|result_option| result_option)
        .flat_map(|(_original_vec_id, neighbor_list)| {
            neighbor_list
                .into_iter()
                .map(|(neighbor_id, score)| SimilarVector {
                    id: neighbor_id,
                    score: score.get_value(),
                })
        })
        .collect();

    if neighbors.is_empty() {
        if !check_vector_existence(ctx, collection_id, vector_id.0).await? {
            return Err(VectorsError::NotFound);
        }
    }

    Ok(neighbors)
}
