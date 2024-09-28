use crate::{
    models::types::{get_app_env, DistanceMetric},
    vector_store::{self, reindex_embeddings},
};

use super::{
    dtos::{DataType, IndexParamsDTo, Quantization},
    error::IndexesError,
};

pub(crate) async fn create_index(
    collection_name: String,
    _name: String,
    distance_metric: DistanceMetric,
    quantization: Quantization,
    data_type: DataType,
    index_params: IndexParamsDTo,
) -> Result<(), IndexesError> {
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => {
            return Err(IndexesError::FailedToGetAppEnv);
        }
    };
    let collection = env
        .vector_store_map
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

    // TODO: trigger re-indexing

    reindex_embeddings(collection).map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;

    Ok(())
}
