use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    config_loader::Config, metadata::MetadataFields, models::types::VectorId, storage::Storage,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWHyperParams {
    pub num_layers: u8,
    pub ef_construction: u32,
    pub ef_search: u32,
    pub max_cache_size: usize,
    pub level_0_neighbors_count: usize,
    pub neighbors_count: usize,
}

impl HNSWHyperParams {
    pub fn default_from_config(config: &Config) -> Self {
        Self {
            ef_construction: config.hnsw.default_ef_construction,
            ef_search: config.hnsw.default_ef_search,
            num_layers: config.hnsw.default_num_layer,
            max_cache_size: config.hnsw.default_max_cache_size,
            level_0_neighbors_count: config.hnsw.default_level_0_neighbors_count,
            neighbors_count: config.hnsw.default_neighbors_count,
        }
    }
}

// Quantized vector embedding
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedDenseVectorEmbedding {
    pub quantized_vec: Arc<Storage>,
    pub hash_vec: VectorId,
}

// Raw vector embedding
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq)]
pub struct RawDenseVectorEmbedding {
    pub raw_vec: Arc<Vec<f32>>,
    pub hash_vec: VectorId,
    pub raw_metadata: Option<MetadataFields>,
    pub is_pseudo: bool,
}
