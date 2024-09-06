use cosdata::storage::Storage;
use serde::{Deserialize, Serialize};

use crate::models::types::DistanceMetric;

#[derive(Debug, Deserialize, Serialize)]
enum QuantizationOptions {
    Binary,
    Quaternay,
    Octal,
    U8,
    F16,
}
#[derive(Debug, Deserialize, Serialize)]
enum Quantization {
    Scalar(QuantizationOptions),
    Product(QuantizationOptions),
}

#[derive(Debug, Deserialize, Serialize)]
struct HSNWHyperParamsDTo {
    m: usize,               // Maximum number of connections per element in the graph default =16
    ef_construction: usize, // Size of the dynamic candidate list during index construction default = 100
    ef_search: usize,       // Size of the dynamic candidate list during search default = 50
    num_layers: usize, // Number of layers in the hierarchical graph (set to 0 for auto-detection)
    max_cache_size: usize, // Maximum number of elements in the cache
    distance_function: DistanceMetric, // Metric for computing distance (e.g., Euclidean, cosine)
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct CreateIndexDto {
    hnsw: HSNWHyperParamsDTo,
    storage: Storage,
    quantization: Quantization,
}
