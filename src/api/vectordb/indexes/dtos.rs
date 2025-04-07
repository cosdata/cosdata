use serde::{Deserialize, Deserializer};

use crate::{
    config_loader::Config, indexes::hnsw::types::HNSWHyperParams, models::types::DistanceMetric,
    quantization::StorageType,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexType {
    Dense,
    Sparse,
}

#[derive(Debug, Default)]
pub enum SparseIndexQuantization {
    #[default]
    B16,
    B32,
    B64,
    B128,
    B256,
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Binary,
    Quaternay,
    Octal,
    U8,
    F16,
    F32,
}

#[derive(Debug, Deserialize)]
pub struct ValuesRange {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct HNSWHyperParamsDto {
    ef_construction: Option<u32>, // Size of the dynamic candidate list during index construction
    ef_search: Option<u32>,       // Size of the dynamic candidate list during search
    num_layers: Option<u8>,       // Number of layers in the hierarchical graph
    max_cache_size: Option<usize>, // Maximum number of elements in the cache
    level_0_neighbors_count: Option<usize>,
    neighbors_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase", tag = "type", content = "properties")]
pub enum DenseIndexParamsDto {
    Hnsw(HNSWHyperParamsDto),
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateDenseIndexDto {
    pub name: String,
    pub distance_metric_type: DistanceMetric,
    pub quantization: DenseIndexQuantizationDto,
    pub index: DenseIndexParamsDto,
}

#[derive(Debug)]
pub(crate) enum CreateSparseIndexDto {
    Splade {
        name: String,
        quantization: SparseIndexQuantization,
        sample_threshold: usize,
    },
    Idf {
        name: String,
    },
}

impl<'de> Deserialize<'de> for CreateSparseIndexDto {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{Error, MapAccess, Visitor};
        use std::fmt;

        struct SparseIndexVisitor;

        impl<'de> Visitor<'de> for SparseIndexVisitor {
            type Value = CreateSparseIndexDto;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a valid sparse index object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut name = None;
                let mut quantization = None;
                let mut sample_threshold = None;
                let mut is_idf = false;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "name" => name = Some(map.next_value()?),
                        "quantization" => quantization = Some(map.next_value()?),
                        "sample_threshold" => sample_threshold = Some(map.next_value()?),
                        "isIDF" => is_idf = map.next_value()?,
                        _ => {
                            return Err(Error::unknown_field(
                                &key,
                                &[
                                    "name",
                                    "quantization",
                                    "early_terminate_threshold",
                                    "sample_threshold",
                                    "isIDF",
                                ],
                            ))
                        }
                    }
                }

                let name = name.ok_or_else(|| Error::missing_field("name"))?;

                if is_idf {
                    Ok(CreateSparseIndexDto::Idf { name })
                } else {
                    Ok(CreateSparseIndexDto::Splade {
                        name,
                        quantization: quantization
                            .ok_or_else(|| Error::missing_field("quantization"))?,
                        sample_threshold: sample_threshold
                            .ok_or_else(|| Error::missing_field("sample_threshold"))?,
                    })
                }
            }
        }

        deserializer.deserialize_map(SparseIndexVisitor)
    }
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
