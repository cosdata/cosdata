use std::sync::Arc;

use crate::app_context::AppContext;

use super::{
    dtos::{CreateDenseIndexDto, CreateSparseIndexDto, CreateTFIDFIndexDto, IndexType},
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
    )
    .await
}

pub(crate) async fn create_tf_idf_index(
    collection_id: String,
    create_index_dto: CreateTFIDFIndexDto,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    repo::create_tf_idf_index(
        ctx,
        collection_id,
        create_index_dto.name,
        create_index_dto.sample_threshold,
        create_index_dto.k1,
        create_index_dto.b,
    )
    .await
}

pub(crate) async fn get_index(
    collection_id: String,
    ctx: Arc<AppContext>,
) -> Result<serde_json::Value, IndexesError> {
    repo::get_index(ctx, collection_id).await
}

pub(crate) async fn delete_index(
    collection_id: String,
    index_type: IndexType,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    repo::delete_index(ctx, collection_id, index_type).await
}
