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
    let collection = ctx
        .ain_env
        .collections_map
        .get(&collection_name)
        .map(|collection| collection.clone())
        .ok_or(IndexesError::CollectionNotFound)?;

    collection.distance_metric.clone().update(distance_metric);
    collection
        .quantization_metric
        .clone()
        .update(quantization.into());
    collection.storage_type.clone().update(data_type.into());

    let IndexParamsDTo::Hnsw(hnsw_hyper_param_dto) = index_params;
    let hnsw_param = hnsw_hyper_param_dto.into();

    collection.hnsw_params.clone().update(hnsw_param);

    create_index_in_collection(collection)
        .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;

    Ok(())
}
