use serde::{Deserialize, Serialize};

use crate::{
    config_loader::Config,
    models::types::{DistanceMetric, HNSWHyperParams},
    quantization::StorageType,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Binary,
    Quaternay,
    Octal,
    U8,
    F16,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ValuesRange {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase", tag = "type", content = "properties")]
pub enum QuantizationDto {
    Auto {
        sample_threshold: usize,
    },
    Scalar {
        data_type: DataType,
        range: ValuesRange,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HNSWHyperParamsDto {
    ef_construction: Option<u32>, // Size of the dynamic candidate list during index construction
    ef_search: Option<u32>,       // Size of the dynamic candidate list during search
    num_layers: Option<u8>,       // Number of layers in the hierarchical graph
    max_cache_size: Option<usize>, // Maximum number of elements in the cache
    level_0_neighbors_count: Option<usize>,
    neighbors_count: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase", tag = "type", content = "properties")]
pub enum IndexParamsDto {
    Hnsw(HNSWHyperParamsDto),
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct CreateIndexDto {
    pub collection_name: String,
    pub name: String,
    pub distance_metric_type: DistanceMetric,
    pub quantization: QuantizationDto,
    pub index: IndexParamsDto,
}

impl HNSWHyperParamsDto {
    pub fn into_params(self, config: &Config) -> HNSWHyperParams {
        let mut default = HNSWHyperParams::default_from_config(config);

        if let Some(ef_construction) = self.ef_construction {
            default.ef_construction = ef_construction;
        }

        if let Some(ef_search) = self.ef_search {
            default.ef_search = ef_search;
        }

        if let Some(num_layers) = self.num_layers {
            default.num_layers = num_layers;
        }

        if let Some(max_cache_size) = self.max_cache_size {
            default.max_cache_size = max_cache_size;
        }

        if let Some(level_0_neighbors_count) = self.level_0_neighbors_count {
            default.level_0_neighbors_count = level_0_neighbors_count;
        }

        if let Some(neighbors_count) = self.neighbors_count {
            default.neighbors_count = neighbors_count;
        }

        default
    }
}

impl From<DataType> for StorageType {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Binary => StorageType::SubByte(1),
            DataType::Quaternay => StorageType::SubByte(2),
            DataType::Octal => StorageType::SubByte(3),
            DataType::U8 => StorageType::UnsignedByte,
            DataType::F16 => StorageType::HalfPrecisionFP,
        }
    }
}
