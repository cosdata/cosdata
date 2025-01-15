use std::sync::Arc;

use crate::{
    api_service::init_dense_index_for_collection,
    app_context::AppContext,
    models::types::{DistanceMetric, QuantizationMetric},
    quantization::StorageType,
};

use super::{
    dtos::{IndexParamsDto, QuantizationDto},
    error::IndexesError,
};

pub(crate) async fn create_index(
    ctx: Arc<AppContext>,
    collection_name: String,
    _name: String,
    distance_metric: DistanceMetric,
    quantization: QuantizationDto,
    index_params: IndexParamsDto,
) -> Result<(), IndexesError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_name)
        .ok_or(IndexesError::CollectionNotFound)?;
    let (quantization_metric, storage_type, range, sample_threshold, is_configured) =
        match quantization {
            QuantizationDto::Auto { sample_threshold } => (
                QuantizationMetric::Scalar,
                StorageType::UnsignedByte,
                None,
                sample_threshold,
                false,
            ),
            QuantizationDto::Scalar { data_type, range } => (
                QuantizationMetric::Scalar,
                data_type.into(),
                Some((range.min, range.max)),
                0,
                true,
            ),
        };
    let IndexParamsDto::Hnsw(hnsw_params_dto) = index_params;
    let hnsw_params = hnsw_params_dto.into_params(&ctx.config);
    init_dense_index_for_collection(
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
