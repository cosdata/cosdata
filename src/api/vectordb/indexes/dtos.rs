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
    Scalar,
    Product,
}

#[derive(Debug, Deserialize, Serialize)]
struct HSNWHyperParamsDTo {
    #[serde(default = "default_m")]
    m: usize, // Maximum number of connections per element in the graph
    #[serde(default = "default_ef_construction")]
    ef_construction: usize, // Size of the dynamic candidate list during index construction
    #[serde(default = "default_ef_search")]
    ef_search: usize, // Size of the dynamic candidate list during search
    #[serde(default)]
    num_layers: usize, // Number of layers in the hierarchical graph
    max_cache_size: usize, // Maximum number of elements in the cache
    distance_function: DistanceMetric, // Metric for computing distance
    quantization: Quantization,
    data_type: QuantizationOptions,
}

fn default_m() -> usize {
    16
}

fn default_ef_construction() -> usize {
    100
}

fn default_ef_search() -> usize {
    50
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct CreateIndexDto {
    hnsw: HSNWHyperParamsDTo,
}
