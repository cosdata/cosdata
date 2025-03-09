use std::sync::Arc;

use crate::app_context::AppContext;

use super::{
    dtos::{CreateDenseIndexDto, CreateSparseIndexDto},
    error::IndexesError,
    repo,
};

pub(crate) async fn create_dense_index(
    collection_id: String,
    create_index_dto: CreateDenseIndexDto,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    repo::create_dense_index(
        ctx,
        collection_id,
        create_index_dto.name,
        create_index_dto.distance_metric_type,
        create_index_dto.quantization,
        create_index_dto.index,
    )
    .await
}

pub(crate) async fn create_sparse_index(
    collection_id: String,
    create_index_dto: CreateSparseIndexDto,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    repo::create_sparse_index(
        ctx,
        collection_id,
        create_index_dto.name,
        create_index_dto.quantization,
        create_index_dto.sample_threshold,
        create_index_dto.search_threshold,
    )
    .await
}
