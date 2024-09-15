use std::sync::Arc;

use crate::{
    api::vectordb::collections,
    api_service::run_upload,
    config_loader::Config,
    models::{rpc::VectorIdValue, types::VectorId},
    vector_store::get_embedding_by_id,
};

use super::{
    dtos::{CreateVectorDto, CreateVectorResponseDto},
    error::VectorsError,
};

pub(crate) async fn create_vector(
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
    config: Arc<Config>,
) -> Result<CreateVectorResponseDto, VectorsError> {
    let collection = collections::service::get_collection_by_id(collection_id)
        .await
        .map_err(|e| VectorsError::FailedToCreateVector(e.to_string()))?;

    // TODO: handle the error
    let _ = run_upload(
        collection,
        vec![(
            create_vector_dto.id.clone(),
            create_vector_dto.values.clone(),
        )],
        config,
    );
    Ok(CreateVectorResponseDto {
        id: create_vector_dto.id,
        values: create_vector_dto.values,
    })
}

pub(crate) async fn get_vector_by_id(
    collection_id: &str,
    vector_id: VectorId,
) -> Result<CreateVectorResponseDto, VectorsError> {
    let vec_store = collections::service::get_collection_by_id(collection_id)
        .await
        .map_err(|_| VectorsError::NotFound)?;

    let embedding = get_embedding_by_id(vec_store, vector_id)
        .map_err(|e| VectorsError::DatabaseError(e.to_string()))?;

    let id = match embedding.hash_vec {
        VectorId::Int(v) => VectorIdValue::IntValue(v),
        VectorId::Str(v) => VectorIdValue::StringValue(v),
    };

    Ok(CreateVectorResponseDto {
        id,
        values: embedding.raw_vec,
    })
}
