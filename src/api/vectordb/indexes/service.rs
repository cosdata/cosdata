use std::sync::Arc;

use crate::app_context::AppContext;

use super::{
    dtos::{CreateDenseIndexDto, CreateSparseIndexDto},
    error::IndexesError,
    repo,
};

pub(crate) async fn create_dense_index(
    create_index_dto: CreateDenseIndexDto,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    repo::create_dense_index(
        ctx,
        create_index_dto.collection_name,
        create_index_dto.name,
        create_index_dto.distance_metric_type,
        create_index_dto.quantization,
        create_index_dto.index,
    )
    .await
}

pub(crate) async fn create_sparse_index(
    create_index_dto: CreateSparseIndexDto,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    repo::create_sparse_index(
        ctx,
        create_index_dto.collection_name,
        create_index_dto.name,
        create_index_dto.quantization,
    )
    .await
}
