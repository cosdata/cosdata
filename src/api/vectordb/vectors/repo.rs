use std::sync::Arc;

use crate::{api::vectordb::collections, api_service::run_upload, config_loader::Config};

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
    _collection_id: &str,
    _vector_id: &str,
) -> Result<CreateVectorResponseDto, VectorsError> {
    Err(VectorsError::NotFound)?
}
