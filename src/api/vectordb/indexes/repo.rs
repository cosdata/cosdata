use std::sync::Arc;

use crate::{
    app_context::AppContext, models::types::DistanceMetric,
    vector_store::create_index_in_collection,
};

use super::{
    dtos::{DataType, IndexParamsDTo, Quantization},
    error::IndexesError,
};

pub(crate) async fn create_index(
    ctx: Arc<AppContext>,
    collection_name: String,
    _name: String,
    distance_metric: DistanceMetric,
    quantization: Quantization,
    data_type: DataType,
    index_params: IndexParamsDTo,
) -> Result<(), IndexesError> {
    let dense_index = ctx
        .ain_env
        .collections_map
        .get(&collection_name)
        .map(|collection| collection.clone())
        .ok_or(IndexesError::CollectionNotFound)?;

    dense_index.distance_metric.clone().update(distance_metric);
    dense_index
        .quantization_metric
        .clone()
        .update(quantization.into());
    dense_index.storage_type.clone().update(data_type.into());

    let IndexParamsDTo::Hnsw(hnsw_hyper_param_dto) = index_params;
    let hnsw_param = hnsw_hyper_param_dto.into();

    dense_index.hnsw_params.clone().update(hnsw_param);

    create_index_in_collection(ctx, dense_index)
        .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;

    Ok(())
}
