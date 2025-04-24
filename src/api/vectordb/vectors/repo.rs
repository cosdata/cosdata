use std::sync::Arc;

use crate::models::{
    collection::Collection, collection_transaction::CollectionTransaction, types::VectorId,
};

use crate::app_context::AppContext;

use super::{
    dtos::{CreateVectorDto, SimilarVector},
    error::VectorsError,
};

pub(crate) fn create_vector_in_transaction(
    ctx: Arc<AppContext>,
    collection: &Collection,
    transaction: &CollectionTransaction,
    create_vector_dto: CreateVectorDto,
) -> Result<(), VectorsError> {
    collection
        .run_upload(vec![create_vector_dto.into()], transaction, &ctx.config)
        .map_err(VectorsError::WaCustom)
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
        .internal_to_external_map
        .get_latest(internal_id)
        .ok_or(VectorsError::NotFound)?
        .clone();
    Ok(vector.into())
}

pub(crate) fn upsert_vectors_in_transaction(
    ctx: Arc<AppContext>,
    collection: &Collection,
    transaction: &CollectionTransaction,
    vectors: Vec<CreateVectorDto>,
) -> Result<(), VectorsError> {
    collection
        .run_upload(
            vectors.into_iter().map(Into::into).collect(),
            transaction,
            &ctx.config,
        )
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
    Ok(collection
        .internal_to_external_map
        .get_latest(internal_id)
        .is_some())
}

pub(crate) async fn fetch_vector_neighbors(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _vector_id: VectorId,
) -> Result<Vec<SimilarVector>, VectorsError> {
    unimplemented!()
}
