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
    match create_index_dto {
        CreateSparseIndexDto::Idf {
            name,
            sample_threshold,
            store_raw_text,
            k1,
            b,
        } => {
            repo::create_sparse_index_idf(
                ctx,
                collection_id,
                name,
                sample_threshold,
                store_raw_text,
                k1,
                b,
            )
            .await
        }
        CreateSparseIndexDto::Splade {
            name,
            quantization,
            sample_threshold,
        } => {
            repo::create_sparse_index(
                ctx,
                collection_id,
                name,
                quantization,
                sample_threshold,
            )
            .await
        }
    }
}

pub(crate) async fn get_index(
    collection_id: String,
    ctx: Arc<AppContext>,
) -> Result<serde_json::Value, IndexesError> {
    repo::get_index(ctx, collection_id).await
}

pub(crate) async fn delete_index(
    collection_id: String,
    index_type: String,
    ctx: Arc<AppContext>,
) -> Result<(), IndexesError> {
    if index_type != "dense" && index_type != "sparse" {
        return Err(IndexesError::InvalidIndexType(index_type));
    }
    repo::delete_index(ctx, collection_id, index_type).await
}
