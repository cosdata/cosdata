use std::sync::Arc;

use crate::{app_context::AppContext, models::types::VectorId};

use super::{
    dtos::{
        CreateDenseVectorDto, CreateVectorDto, CreateVectorResponseDto, UpdateVectorDto,
        UpdateVectorResponseDto, SimilarVector
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
        CreateVectorDto::SparseIdf(dto) => {
            repo::create_sparse_idf_document(ctx, collection_id, dto).await
        }
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


pub(crate) async fn delete_vector_by_id(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: u64,
) -> Result<(), VectorsError> {
    repo::delete_vector_by_id(ctx, collection_id, vector_id).await
}

pub(crate) async fn check_vector_existence(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: u64,
) -> Result<bool, VectorsError> {
    repo::check_vector_existence(ctx, collection_id, vector_id).await
}


pub(crate) async fn fetch_vector_neighbors(
    ctx: Arc<AppContext>,
    collection_id: &str,
    vector_id: VectorId,
) -> Result<Vec<SimilarVector>, VectorsError> {
    repo::fetch_vector_neighbors(ctx, collection_id, vector_id).await
}
