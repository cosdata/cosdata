use std::sync::Arc;

use crate::{
    indexes::{
        hnsw::DenseInputEmbedding, inverted::SparseInputEmbedding, tf_idf::TFIDFInputEmbedding,
        IndexOps,
    },
    models::{
        collection::Collection, collection_transaction::CollectionTransaction, types::VectorId,
    },
};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateVectorDto, SimilarVector},
    error::VectorsError,
};

pub(crate) async fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection: &Collection,
    transaction: &CollectionTransaction,
    create_vector_dto: CreateVectorDto,
) -> Result<(), VectorsError> {
    if let Some(values) = create_vector_dto.dense_values {
        let Some(hnsw_index) = collection.get_hnsw_index() else {
            return Err(VectorsError::IndexNotFound);
        };
        hnsw_index
            .run_upload(
                &collection,
                vec![DenseInputEmbedding(
                    create_vector_dto.id.clone(),
                    values,
                    create_vector_dto.metadata,
                    false,
                )],
                transaction,
                &ctx.config,
            )
            .map_err(VectorsError::WaCustom)?;
    }
    if let Some(values) = create_vector_dto.sparse_values {
        let Some(inverted_index) = collection.get_inverted_index() else {
            return Err(VectorsError::IndexNotFound);
        };
        inverted_index
            .run_upload(
                &collection,
                vec![SparseInputEmbedding(create_vector_dto.id.clone(), values)],
                transaction,
                &ctx.config,
            )
            .map_err(VectorsError::WaCustom)?;
    }
    if let Some(text) = create_vector_dto.text {
        let Some(tf_idf_index) = collection.get_tf_idf_index() else {
            return Err(VectorsError::IndexNotFound);
        };
        tf_idf_index
            .run_upload(
                &collection,
                vec![TFIDFInputEmbedding(create_vector_dto.id, text)],
                transaction,
                &ctx.config,
            )
            .map_err(VectorsError::WaCustom)?;
    }

    Ok(())
}

pub(crate) async fn get_vector_by_id(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _vector_id: VectorId,
) -> Result<CreateVectorDto, VectorsError> {
    unimplemented!()
}

pub(crate) async fn upsert_vectors_in_transaction(
    ctx: Arc<AppContext>,
    collection: &Collection,
    transaction: &CollectionTransaction,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), VectorsError> {
    let (dense_vec, sparse_vec, tf_idf_vec): (Vec<_>, Vec<_>, Vec<_>) =
        vectors
            .into_iter()
            .fold((Vec::new(), Vec::new(), Vec::new()), |mut acc, dto| {
                let CreateVectorDto {
                    id,
                    dense_values,
                    metadata,
                    sparse_values,
                    text,
                } = dto;

                if let Some(values) = dense_values {
                    acc.0.push(DenseInputEmbedding(id, values, metadata, false));
                } else if let Some(values) = sparse_values {
                    acc.1.push(SparseInputEmbedding(id, values));
                } else if let Some(text) = text {
                    acc.2.push(TFIDFInputEmbedding(id, text));
                }

                acc
            });

    if !dense_vec.is_empty() {
        let Some(hnsw_index) = collection.get_hnsw_index() else {
            return Err(VectorsError::IndexNotFound);
        };

        hnsw_index
            .run_upload(collection, dense_vec, transaction, &ctx.config)
            .map_err(VectorsError::WaCustom)?;
    }

    if !sparse_vec.is_empty() {
        let Some(inverted_index) = collection.get_inverted_index() else {
            return Err(VectorsError::IndexNotFound);
        };

        inverted_index
            .run_upload(collection, sparse_vec, transaction, &ctx.config)
            .map_err(VectorsError::WaCustom)?;
    }

    if !tf_idf_vec.is_empty() {
        let Some(tf_idf_index) = collection.get_tf_idf_index() else {
            return Err(VectorsError::IndexNotFound);
        };

        tf_idf_index
            .run_upload(collection, tf_idf_vec, transaction, &ctx.config)
            .map_err(VectorsError::WaCustom)?;
    }

    Ok(())
}

pub(crate) async fn check_vector_existence(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _vector_id: u64,
) -> Result<bool, VectorsError> {
    unimplemented!()
}

pub(crate) async fn fetch_vector_neighbors(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _vector_id: VectorId,
) -> Result<Vec<SimilarVector>, VectorsError> {
    unimplemented!()
}
