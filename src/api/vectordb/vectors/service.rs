use std::sync::Arc;

use crate::{config_loader::Config, models::rpc::VectorIdValue};

use super::{
    dtos::{
        CreateVectorDto, CreateVectorResponseDto, FindSimilarVectorsResponseDto, UpdateVectorDto,
        UpdateVectorResponseDto,
    },
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
    vector_id: &str,
) -> Result<CreateVectorResponseDto, VectorsError> {
    repo::get_vector_by_id(collection_id, vector_id).await
}

pub(crate) async fn update_vector_by_id(
    collection_id: &str,
    vector_id: VectorIdValue,
    update_vector_dto: UpdateVectorDto,
    config: Arc<Config>,
) -> Result<UpdateVectorResponseDto, VectorsError> {
    repo::update_vector(collection_id, vector_id, update_vector_dto, config).await
}

pub(crate) async fn find_similar_vectors() -> Result<FindSimilarVectorsResponseDto, VectorsError> {
    let similar_vectors = repo::find_similar_vectors().await?;

    Ok(FindSimilarVectorsResponseDto {
        results: similar_vectors,
    })
}
