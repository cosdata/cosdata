use std::sync::Arc;

use crate::{config_loader::Config, models::types::VectorId};

use super::{
    dtos::{CreateVectorDto, CreateVectorResponseDto},
    error::VectorsError,
    repo,
};

pub(crate) async fn create_vector(
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
    config: Arc<Config>,
) -> Result<CreateVectorResponseDto, VectorsError> {
    repo::create_vector(collection_id, create_vector_dto, config).await
}

pub(crate) async fn get_vector_by_id(
    collection_id: &str,
    vector_id: VectorId,
) -> Result<CreateVectorResponseDto, VectorsError> {
    repo::get_vector_by_id(collection_id, vector_id).await
}
