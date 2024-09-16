use std::sync::Arc;

use crate::{
    api::vectordb::collections, api_service::run_upload, config_loader::Config,
    models::rpc::VectorIdValue,
};

use super::{
    dtos::{CreateVectorDto, CreateVectorResponseDto, UpdateVectorDto, UpdateVectorResponseDto},
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

    // error cases that happens within run_upload is not handled
    // this method always return a successful response with the data sent by the user
    run_upload(
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

pub(crate) async fn update_vector(
    collection_id: &str,
    vector_id: VectorIdValue,
    update_vector_dto: UpdateVectorDto,
    config: Arc<Config>,
) -> Result<UpdateVectorResponseDto, VectorsError> {
    let collection = collections::service::get_collection_by_id(collection_id)
        .await
        .map_err(|e| VectorsError::FailedToUpdateVector(e.to_string()))?;

    // error cases that happens within run_upload is not handled
    // this method always return a successful response with the data sent by the user
    run_upload(
        collection,
        vec![(vector_id.clone(), update_vector_dto.values.clone())],
        config,
    );
    Ok(UpdateVectorResponseDto {
        id: vector_id,
        values: update_vector_dto.values,
    })
}

pub(crate) async fn delete_vector_by_id(
    _collection_id: &str,
    _vector_id: VectorIdValue,
) -> Result<CreateVectorResponseDto, VectorsError> {
    Err(VectorsError::NotImplemented)?
}
