use std::sync::Arc;

use crate::{app_context::AppContext, models::types::VectorId};

use super::{
    dtos::{
        BatchSearchVectorsDto, CreateDenseVectorDto, CreateVectorDto, CreateVectorResponseDto,
        FindSimilarVectorsDto, FindSimilarVectorsResponseDto, UpdateVectorDto,
        UpdateVectorResponseDto,
    },
    error::VectorsError,
    repo,
};

pub(crate) async fn create_vector(
    ctx: Arc<AppContext>,
    collection_id: &str,
    create_vector_dto: CreateVectorDto,
) -> Result<CreateVectorResponseDto, VectorsError> {
    match create_vector_dto {
        CreateVectorDto::Dense(dto) => repo::create_dense_vector(ctx, collection_id, dto).await,
        CreateVectorDto::Sparse(dto) => repo::create_sparse_vector(ctx, collection_id, dto).await,
    }
}

pub(crate) async fn get_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<CreateDenseVectorDto, VectorsError> {
    repo::get_vector_by_id(ctx, collection_id, vector_id).await
}

pub(crate) async fn update_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
    update_vector_dto: UpdateVectorDto,
) -> Result<UpdateVectorResponseDto, VectorsError> {
    repo::update_vector(ctx, collection_id, vector_id, update_vector_dto).await
}

pub(crate) async fn find_similar_vectors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    find_similar_vectors: FindSimilarVectorsDto,
) -> Result<FindSimilarVectorsResponseDto, VectorsError> {
    match find_similar_vectors {
        FindSimilarVectorsDto::Dense(dto) => {
            repo::find_similar_dense_vectors(ctx, collection_id, dto).await
        }
        FindSimilarVectorsDto::Sparse(dto) => {
            repo::find_similar_sparse_vectors(ctx, collection_id, dto).await
        }
    }
}

pub(crate) async fn batch_search(
    ctx: Arc<AppContext>,
    collection_id: &str,
    batch_search_vectors: BatchSearchVectorsDto,
) -> Result<Vec<FindSimilarVectorsResponseDto>, VectorsError> {
    match batch_search_vectors {
        BatchSearchVectorsDto::Dense(dto) => {
            repo::batch_search_dense_vectors(ctx, collection_id, dto).await
        }
        BatchSearchVectorsDto::Sparse(dto) => {
            repo::batch_search_sparse_vectors(ctx, collection_id, dto).await
        }
    }
}

pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: u64,
) -> Result<(), VectorsError> {
    repo::delete_vector_by_id(ctx, collection_id, vector_id).await
}
