use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    api_service::{
        init_geozone_index_for_collection, init_hnsw_index_for_collection,
        init_inverted_index_for_collection, init_tf_idf_index_for_collection,
    },
    app_context::AppContext,
    models::types::{DistanceMetric, QuantizationMetric},
    quantization::StorageType,
};

use super::{
    dtos::{DenseIndexParamsDto, DenseIndexQuantizationDto, IndexType, SparseIndexQuantization},
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

    // Check if index already exists BEFORE initializing
    if collection.get_hnsw_index().is_some() {
        return Err(IndexesError::IndexAlreadyExists("dense".to_string()));
    }

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
        collection,
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
        .ok_or_else(|| {
            IndexesError::NotFound(format!("Collection '{}' not found", collection_name))
        })?;

    if collection.meta.sparse_vector.geofencing {
        if collection.get_geozone_index().is_some() {
            return Err(IndexesError::IndexAlreadyExists("sparse".to_string()));
        }

        init_geozone_index_for_collection(ctx.clone(), &collection, quantization.into_bits())
            .await
            .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;
    } else {
        if collection.get_inverted_index().is_some() {
            return Err(IndexesError::IndexAlreadyExists("sparse".to_string()));
        }

        init_inverted_index_for_collection(
            ctx.clone(),
            &collection,
            quantization.into_bits(),
            sample_threshold,
        )
        .await
        .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;
    }

    Ok(())
}

pub(crate) async fn create_tf_idf_index(
    ctx: Arc<AppContext>,
    collection_name: String,
    _name: String,
    sample_threshold: usize,
    k1: f32,
    b: f32,
) -> Result<(), IndexesError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_name)
        .ok_or(IndexesError::CollectionNotFound)?;

    if collection.get_tf_idf_index().is_some() {
        return Err(IndexesError::IndexAlreadyExists("tf_idf".to_string()));
    }

    init_tf_idf_index_for_collection(ctx, &collection, sample_threshold, k1, b)
        .await
        .map_err(|e| IndexesError::FailedToCreateIndex(e.to_string()))?;

    Ok(())
}

pub(crate) async fn get_index(
    ctx: Arc<AppContext>,
    collection_name: String,
) -> Result<serde_json::Value, IndexesError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_name)
        .ok_or_else(|| {
            IndexesError::NotFound(format!("Collection '{}' not found", collection_name))
        })?;

    let mut indexes_array = Vec::new();

    if let Some(hnsw) = collection.get_hnsw_index() {
        let distance_metric = *hnsw.distance_metric.read().unwrap();
        let values_range = *hnsw.values_range.read().unwrap();
        let hnsw_params = hnsw.hnsw_params.read().unwrap();
        let storage_type = *hnsw.storage_type.read().unwrap();
        let quantization = format!("{:?}", *hnsw.quantization_metric.read().unwrap());

        indexes_array.push(serde_json::json!({
            "type": "dense",
            "name": collection_name,
            "algorithm": "HNSW",
            "distance_metric": format!("{:?}", distance_metric),
            "quantization": {
                "type": quantization,
                "storage": format!("{:?}", storage_type),
                "range": { "min": values_range.0, "max": values_range.1 }
            },
            "params": {
                "ef_construction": hnsw_params.ef_construction,
                "ef_search": hnsw_params.ef_search,
                "neighbors_count": hnsw_params.neighbors_count,
                "level_0_neighbors_count": hnsw_params.level_0_neighbors_count,
                "num_layers": hnsw_params.num_layers,
            }
        }));
    }

    if let Some(inverted) = collection.get_inverted_index() {
        let values_upper_bound = *inverted.values_upper_bound.read().unwrap();
        indexes_array.push(serde_json::json!({
            "type": "sparse",
            "name": collection_name,
            "algorithm": "InvertedIndex",
            "quantization_bits": inverted.root.root.quantization_bits,
            "values_upper_bound": values_upper_bound,
        }));
    }

    Ok(serde_json::json!({
        "collection_name": collection_name,
        "indexes": indexes_array
    }))
}

pub(crate) async fn delete_index(
    ctx: Arc<AppContext>,
    collection_name: String,
    index_type: IndexType,
) -> Result<(), IndexesError> {
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(&collection_name)
        .ok_or_else(|| {
            IndexesError::NotFound(format!("Collection '{}' not found", collection_name))
        })?;

    let collection_path: PathBuf = collection.get_path().to_path_buf();

    match index_type {
        IndexType::Dense => {
            if collection.get_hnsw_index().is_none() {
                return Err(IndexesError::NotFound(format!(
                    "Dense index does not exist for collection '{}'",
                    collection_name
                )));
            }
            ctx.ain_env
                .collections_map
                .remove_hnsw_index(&collection_name)
                .map_err(|e| {
                    IndexesError::FailedToDeleteIndex(format!(
                        "Failed to remove dense index from map: {}",
                        e
                    ))
                })?;

            let index_path = collection_path.join("dense_hnsw");
            if index_path.exists() {
                log::info!(
                    "Attempting to remove dense index directory: {:?}",
                    index_path
                );
                fs::remove_dir_all(&index_path).map_err(|e| {
                    IndexesError::FailedToDeleteIndex(format!(
                        "Failed to remove dense index directory '{}': {}",
                        index_path.display(),
                        e
                    ))
                })?;
                log::info!(
                    "Successfully removed dense index directory: {:?}",
                    index_path
                );
            } else {
                log::warn!(
                    "Dense index directory not found for removal: {:?}",
                    index_path
                );
            }
            log::info!(
                "Dense index removed successfully for collection '{}'",
                collection_name
            );
        }
        IndexType::Sparse => {
            if collection.get_inverted_index().is_none() {
                return Err(IndexesError::NotFound(format!(
                    "Sparse index does not exist for collection '{}'",
                    collection_name
                )));
            }
            ctx.ain_env
                .collections_map
                .remove_inverted_index(&collection_name)
                .map_err(|e| {
                    IndexesError::FailedToDeleteIndex(format!(
                        "Failed to remove sparse index from map: {}",
                        e
                    ))
                })?;

            let index_path = collection_path.join("sparse_inverted_index");
            if index_path.exists() {
                log::info!(
                    "Attempting to remove sparse index directory: {:?}",
                    index_path
                );
                fs::remove_dir_all(&index_path).map_err(|e| {
                    IndexesError::FailedToDeleteIndex(format!(
                        "Failed to remove sparse index directory '{}': {}",
                        index_path.display(),
                        e
                    ))
                })?;
                log::info!(
                    "Successfully removed sparse index directory: {:?}",
                    index_path
                );
            } else {
                log::warn!(
                    "Sparse index directory not found for removal: {:?}",
                    index_path
                );
            }
            log::info!(
                "Sparse index removed successfully for collection '{}'",
                collection_name
            );
        }
        IndexType::TfIdf => {
            if collection.get_tf_idf_index().is_none() {
                return Err(IndexesError::NotFound(format!(
                    "TF-IDF index does not exist for collection '{}'",
                    collection_name
                )));
            }
            ctx.ain_env
                .collections_map
                .remove_tf_idf_index(&collection_name)
                .map_err(|e| {
                    IndexesError::FailedToDeleteIndex(format!(
                        "Failed to remove TF-IDF index from map: {}",
                        e
                    ))
                })?;

            let index_path = collection_path.join("tf_idf_index");
            if index_path.exists() {
                log::info!(
                    "Attempting to remove TF-IDF index directory: {:?}",
                    index_path
                );
                fs::remove_dir_all(&index_path).map_err(|e| {
                    IndexesError::FailedToDeleteIndex(format!(
                        "Failed to remove TF-IDF index directory '{}': {}",
                        index_path.display(),
                        e
                    ))
                })?;
                log::info!(
                    "Successfully removed TF-IDF index directory: {:?}",
                    index_path
                );
            } else {
                log::warn!(
                    "TF-IDF index directory not found for removal: {:?}",
                    index_path
                );
            }
            log::info!(
                "TF-IDF index removed successfully for collection '{}'",
                collection_name
            );
        }
    }

    // TODO: Consider if associated LMDB entries specific to the index need cleanup.
    // Currently, only the directory and the in-memory map entry are removed.

    Ok(())
}
