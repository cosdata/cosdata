use std::sync::Arc;

use crate::{app_context::AppContext, models::types::VectorId};

use super::{
    dtos::{
        CreateVectorDto, CreateVectorResponseDto, FindSimilarVectorsDto,
        FindSimilarVectorsResponseDto, UpdateVectorDto, UpdateVectorResponseDto,
    },
    error::VectorsError,
    repo,
};

pub(crate) async fn create_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
    repo::create_vector(ctx, collection_id, create_vector_dto).await
}

pub(crate) async fn get_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<CreateVectorResponseDto, VectorsError> {
    repo::get_vector_by_id(ctx, collection_id, vector_id).await
}

pub(crate) async fn update_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: u64,
    update_vector_dto: UpdateVectorDto,
) -> Result<UpdateVectorResponseDto, VectorsError> {
    repo::update_vector(ctx, collection_id, vector_id, update_vector_dto).await
}

pub(crate) async fn find_similar_vectors(
    find_similar_vectors: FindSimilarVectorsDto,
) -> Result<FindSimilarVectorsResponseDto, VectorsError> {
    let similar_vectors = repo::find_similar_vectors(find_similar_vectors).await?;

    Ok(FindSimilarVectorsResponseDto {
        results: similar_vectors,
    })
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: u64,
) -> Result<(), VectorsError> {
    repo::delete_vector_by_id(ctx, collection_id, vector_id).await
}
