use std::sync::Arc;

use crate::models::types::DocumentId;
use crate::models::{
    collection::Collection, collection_transaction::ExplicitTransaction, types::VectorId,
};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateVectorDto, SimilarVector},
    error::VectorsError,
};

pub(crate) fn create_vector_in_transaction(
    collection: &Collection,
    transaction: &ExplicitTransaction,
    create_vector_dto: CreateVectorDto,
) -> Result<(), VectorsError> {
    collection
        .run_upload(vec![create_vector_dto.into()], transaction)
        .map_err(VectorsError::WaCustom)
}

pub(crate) async fn query_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    document_id: DocumentId,
) -> Result<Vec<CreateVectorDto>, VectorsError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(VectorsError::CollectionNotFound)?;

    let Some(internal_ids) = collection.document_to_internals_map.get(&document_id) else {
        return Ok(Vec::new());
    };

    internal_ids
        .iter()
        .map(|internal_id| {
            Ok(collection
                .get_raw_emb_by_internal_id(&internal_id)
                .ok_or(VectorsError::NotFound)?
                .clone()
                .into())
        })
        .collect()
}

pub(crate) async fn get_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<CreateVectorDto, VectorsError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(VectorsError::CollectionNotFound)?;
    let internal_id = collection
        .external_to_internal_map
        .get_latest(&vector_id)
        .ok_or(VectorsError::NotFound)?;
    let vector = collection
        .get_raw_emb_by_internal_id(internal_id)
        .ok_or(VectorsError::NotFound)?
        .clone();
    Ok(vector.into())
}

pub(crate) fn upsert_vectors_in_transaction(
    collection: &Collection,
    transaction: &ExplicitTransaction,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), VectorsError> {
    collection
        .run_upload(vectors.into_iter().map(Into::into).collect(), transaction)
        .map_err(VectorsError::WaCustom)
}

pub(crate) async fn check_vector_existence(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<bool, VectorsError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or(VectorsError::CollectionNotFound)?;
    let internal_id =
        if let Some(internal_id) = collection.external_to_internal_map.get_latest(&vector_id) {
            internal_id
        } else {
            return Ok(false);
        };
    Ok(collection.get_raw_emb_by_internal_id(internal_id).is_some())
}

pub(crate) async fn fetch_vector_neighbors(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _vector_id: VectorId,
) -> Result<Vec<SimilarVector>, VectorsError> {
    unimplemented!()
}
