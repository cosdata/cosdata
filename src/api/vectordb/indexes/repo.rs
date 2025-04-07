use std::sync::Arc;

use crate::{
    api_service::{
        init_hnsw_index_for_collection, init_inverted_index_for_collection,
        init_inverted_index_idf_for_collection,
    },
    app_context::AppContext,
    models::types::{DistanceMetric, QuantizationMetric},
    quantization::StorageType,
};

use super::{
    dtos::{DenseIndexParamsDto, DenseIndexQuantizationDto, SparseIndexQuantization},
    error::IndexesError,
};

pub(crate) async fn create_dense_index(
    ctx: Arc<AppContext>,
    collection_name: String,
    _name: String,
    distance_metric: DistanceMetric,
    quantization: DenseIndexQuantizationDto,
    index_params: DenseIndexParamsDto,
) -> Result<(), IndexesError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_name)
        .ok_or(IndexesError::CollectionNotFound)?;
    let (quantization_metric, storage_type, range, sample_threshold, is_configured) =
        match quantization {
            DenseIndexQuantizationDto::Auto { sample_threshold } => (
                QuantizationMetric::Scalar,
                StorageType::UnsignedByte,
                None,
                sample_threshold,
                false,
            ),
            DenseIndexQuantizationDto::Scalar { data_type, range } => (
                QuantizationMetric::Scalar,
                data_type.into(),
                Some((range.min, range.max)),
                0,
                true,
            ),
        };
    let DenseIndexParamsDto::Hnsw(hnsw_params_dto) = index_params;
    let hnsw_params = hnsw_params_dto.into_params(&ctx.config);
    init_hnsw_index_for_collection(
        ctx,
        &collection,
        range,
        hnsw_params,
        quantization_metric,
        distance_metric,
        storage_type,
        sample_threshold,
        is_configured,
    )
    .await
    .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;

    Ok(())
}

pub(crate) async fn create_sparse_index(
    ctx: Arc<AppContext>,
    collection_name: String,
    _name: String,
    quantization: SparseIndexQuantization,
    sample_threshold: usize,
) -> Result<(), IndexesError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_name)
        .ok_or(IndexesError::CollectionNotFound)?;

    init_inverted_index_for_collection(
        ctx,
        &collection,
        quantization.into_bits(),
        sample_threshold,
    )
    .await
    .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;

    Ok(())
}

pub(crate) async fn create_sparse_index_idf(
    ctx: Arc<AppContext>,
    collection_name: String,
    _name: String,
    quantization: SparseIndexQuantization,
) -> Result<(), IndexesError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_name)
        .ok_or(IndexesError::CollectionNotFound)?;

    init_inverted_index_idf_for_collection(
        ctx,
        &collection,
        quantization.into_bits(),
    )
    .await
    .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;

    Ok(())
}
