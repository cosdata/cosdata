use serde::{Deserialize, Deserializer, Serialize};
use utoipa::ToSchema;

use crate::{
    config_loader::Config, indexes::hnsw::types::HNSWHyperParams, 
    models::schema_traits::DistanceMetricSchema, quantization::StorageType,
};

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    Dense,
    Sparse,
    TfIdf,
}

#[derive(Debug, Default, ToSchema)]
pub enum SparseIndexQuantization {
    #[default]
    B16,
    B32,
    B64,
    B128,
    B256,
}

// Response DTOs for indexes
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct IndexResponseDto {
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct IndexDetailsDto {
    pub collection_name: String,
    pub indexes: Vec<IndexInfo>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub enum IndexInfo {
    #[serde(rename = "dense")]
    Dense(DenseIndexInfo),
    #[serde(rename = "sparse")]
    Sparse(SparseIndexInfo),
    #[serde(rename = "tf_idf")]
    TfIdf(TfIdfIndexInfo),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DenseIndexInfo {
    pub name: String,
    pub algorithm: String,
    pub distance_metric: String,
    pub quantization: QuantizationInfo,
    pub params: HnswParamsInfo,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QuantizationInfo {
    #[serde(rename = "type")]
    pub quantization_type: String,
    pub storage: String,
    pub range: RangeInfo,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RangeInfo {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HnswParamsInfo {
    pub ef_construction: u32,
    pub ef_search: u32,
    pub neighbors_count: usize,
    pub level_0_neighbors_count: usize,
    pub num_layers: u8,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SparseIndexInfo {
    pub name: String,
    pub algorithm: String,
    pub quantization_bits: u8,
    pub values_upper_bound: f32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TfIdfIndexInfo {
    pub name: String,
    pub algorithm: String,
    pub k1: f32,
    pub b: f32,
}

impl<'de> Deserialize<'de> for SparseIndexQuantization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: u64 = Deserialize::deserialize(deserializer)?;
        match value {
            16 => Ok(Self::B16),
            32 => Ok(Self::B32),
            64 => Ok(Self::B64),
            128 => Ok(Self::B128),
            256 => Ok(Self::B256),
            _ => Err(serde::de::Error::custom(format!(
                "Invalid value for quantization: {}. Expected 16, 32, 64, 128 or 256.",
                value
            ))),
        }
    }
}

impl SparseIndexQuantization {
    pub fn into_bits(self) -> u8 {
        match self {
            Self::B16 => 4,
            Self::B32 => 5,
            Self::B64 => 6,
            Self::B128 => 7,
            Self::B256 => 8,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Binary,
    Quaternay,
    Octal,
    U8,
    F16,
    F32,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ValuesRange {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase", tag = "type", content = "properties")]
pub enum DenseIndexQuantizationDto {
    Auto {
        sample_threshold: usize,
    },
    Scalar {
        data_type: DataType,
        range: ValuesRange,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct HNSWHyperParamsDto {
    ef_construction: Option<u32>, // Size of the dynamic candidate list during index construction
    ef_search: Option<u32>,       // Size of the dynamic candidate list during search
    num_layers: Option<u8>,       // Number of layers in the hierarchical graph
    max_cache_size: Option<usize>, // Maximum number of elements in the cache
    level_0_neighbors_count: Option<usize>,
    neighbors_count: Option<usize>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase", tag = "type", content = "properties")]
pub enum DenseIndexParamsDto {
    Hnsw(HNSWHyperParamsDto),
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateDenseIndexDto {
    pub name: String,
    pub distance_metric_type: DistanceMetricSchema,
    pub quantization: DenseIndexQuantizationDto,
    pub index: DenseIndexParamsDto,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateSparseIndexDto {
    pub name: String,
    pub quantization: SparseIndexQuantization,
    pub sample_threshold: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateTFIDFIndexDto {
    pub name: String,
    pub sample_threshold: usize,
    pub k1: f32,
    pub b: f32,
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
            DataType::F32 => StorageType::FullPrecisionFP,
        }
    }
}
