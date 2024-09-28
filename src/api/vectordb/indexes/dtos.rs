use serde::{Deserialize, Serialize};

use crate::{
    models::types::{DistanceMetric, HNSWHyperParams, QuantizationMetric},
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
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    Scalar,
    Product,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HNSWHyperParamsDTo {
    #[serde(default = "default_m")]
    m: usize, // Maximum number of connections per element in the graph
    #[serde(default = "default_ef_construction")]
    ef_construction: usize, // Size of the dynamic candidate list during index construction
    #[serde(default = "default_ef_search")]
    ef_search: usize, // Size of the dynamic candidate list during search
    #[serde(default = "default_num_layers")]
    num_layers: u8, // Number of layers in the hierarchical graph
    max_cache_size: usize, // Maximum number of elements in the cache
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

fn default_num_layers() -> u8 {
    5
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase", tag = "index_type", content = "params")]
pub enum IndexParamsDTo {
    Hnsw(HNSWHyperParamsDTo),
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct CreateIndexDto {
    pub collection_name: String,
    pub name: String,
    pub distance_metric_type: DistanceMetric,
    pub quantization: Quantization,
    pub data_type: DataType,
    #[serde(flatten)]
    pub index_params: IndexParamsDTo,
}

impl From<HNSWHyperParamsDTo> for HNSWHyperParams {
    fn from(params: HNSWHyperParamsDTo) -> Self {
        Self {
            m: params.m,
            ef_construction: params.ef_construction,
            ef_search: params.ef_search,
            num_layers: params.num_layers,
            max_cache_size: params.max_cache_size,
        }
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

impl From<Quantization> for QuantizationMetric {
    fn from(quantization: Quantization) -> Self {
        match quantization {
            Quantization::Scalar => Self::Scalar,
            Quantization::Product => Self::Product(Default::default()),
        }
    }
}
