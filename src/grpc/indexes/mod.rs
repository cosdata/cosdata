use std::sync::Arc;
use tonic::{Request, Response, Status};
use prost::Message;

use crate::app_context::AppContext;
use crate::models::common::WaCustomError;

crate::cfg_grpc! {
use super::proto::indexes_service_server::IndexesService;
use super::proto::{
    CreateDenseIndexRequest,
    CreateSparseIndexRequest,
    DataType as ProtoDataType,
    ValuesRange,
    HnswParams,
};
use super::proto::create_dense_index_request;

use crate::api::vectordb::indexes::service;
use crate::api::vectordb::indexes::dtos::{
    CreateDenseIndexDto,
    CreateSparseIndexDto,
    DenseIndexQuantizationDto,
    DenseIndexParamsDto,
    DataType,
    ValuesRange as DtoValuesRange,
    HNSWHyperParamsDto,
    SparseIndexQuantization,
};
use crate::models::types::DistanceMetric;

pub struct IndexesServiceImpl {
    pub context: Arc<AppContext>,
}

#[tonic::async_trait]
impl IndexesService for IndexesServiceImpl {
    async fn create_dense_index(
        &self,
        request: Request<CreateDenseIndexRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        // Parse distance metric
        let distance_metric = match req.distance_metric_type.as_str() {
            "cosine" => DistanceMetric::Cosine,
            "dotproduct" => DistanceMetric::DotProduct,
            "euclidean" => DistanceMetric::Euclidean,
            _ => return Err(Status::invalid_argument("Invalid distance metric type")),
        };

        // Parse quantization
        let quantization = match req.quantization {
            Some(create_dense_index_request::Quantization::Auto(auto)) => {
                DenseIndexQuantizationDto::Auto {
                    sample_threshold: auto.sample_threshold as usize,
                }
            }
            Some(create_dense_index_request::Quantization::Scalar(scalar)) => {
                // Convert data type
                let data_type = match ProtoDataType::from_i32(scalar.data_type).ok_or_else(|| {
                    Status::invalid_argument("Invalid data type")
                })? {
                    ProtoDataType::Binary => DataType::Binary,
                    ProtoDataType::Quaternary => DataType::Quaternay,
                    ProtoDataType::Octal => DataType::Octal,
                    ProtoDataType::U8 => DataType::U8,
                    ProtoDataType::F16 => DataType::F16,
                    ProtoDataType::F32 => DataType::F32,
                };

                // Extract range
                let range = scalar.range.ok_or_else(|| {
                    Status::invalid_argument("Range is required for scalar quantization")
                })?;

                DenseIndexQuantizationDto::Scalar {
                    data_type,
                    range: DtoValuesRange {
                        min: range.min,
                        max: range.max,
                    }
                }
            }
            None => return Err(Status::invalid_argument("Quantization is required")),
        };

        // Parse HNSW parameters
        let hnsw_params = req.hnsw_params.ok_or_else(|| {
            Status::invalid_argument("HNSW parameters are required")
        })?;

        use serde_json::{json, Value};

        let mut hnsw_json = json!({
            "type": "hnsw",
            "properties": {}
        });

        if let Some(props) = hnsw_json.get_mut("properties").and_then(|v| v.as_object_mut()) {
            if let Some(ef_construction) = hnsw_params.ef_construction {
                props.insert("ef_construction".to_string(), json!(ef_construction));
            }
            if let Some(ef_search) = hnsw_params.ef_search {
                props.insert("ef_search".to_string(), json!(ef_search));
            }
            if let Some(num_layers) = hnsw_params.num_layers {
                props.insert("num_layers".to_string(), json!(num_layers));
            }
            if let Some(max_cache_size) = hnsw_params.max_cache_size {
                props.insert("max_cache_size".to_string(), json!(max_cache_size));
            }
            if let Some(level_0_neighbors_count) = hnsw_params.level_0_neighbors_count {
                props.insert("level_0_neighbors_count".to_string(), json!(level_0_neighbors_count));
            }
            if let Some(neighbors_count) = hnsw_params.neighbors_count {
                props.insert("neighbors_count".to_string(), json!(neighbors_count));
            }
        }

        // Deserialize the JSON into DenseIndexParamsDto
        let index_params: DenseIndexParamsDto = serde_json::from_value(hnsw_json)
            .map_err(|e| Status::internal(format!("Failed to create index parameters: {}", e)))?;

        // Create DTO
        let create_index_dto = CreateDenseIndexDto {
            name: req.name,
            distance_metric_type: distance_metric,
            quantization,
            index: index_params,
        };

        // Call service
        service::create_dense_index(
            req.collection_id,
            create_index_dto,
            self.context.clone(),
        )
        .await
        .map_err(|e| Status::internal(format!("Failed to create dense index: {}", e)))?;

        Ok(Response::new(()))
    }

    async fn create_sparse_index(
        &self,
        request: Request<CreateSparseIndexRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        // Parse quantization (convert from u32 to SparseIndexQuantization)
        let quantization = match req.quantization {
            16 => SparseIndexQuantization::B16,
            32 => SparseIndexQuantization::B32,
            64 => SparseIndexQuantization::B64,
            128 => SparseIndexQuantization::B128,
            256 => SparseIndexQuantization::B256,
            _ => return Err(Status::invalid_argument(
                "Invalid quantization value. Expected 16, 32, 64, 128, or 256."
            )),
        };

        // Create DTO
        let create_index_dto = CreateSparseIndexDto {
            name: req.name,
            quantization,
            sample_threshold: req.quantization as usize, // Using the same value for sample threshold
        };

        // Call service
        service::create_sparse_index(
            req.collection_id,
            create_index_dto,
            self.context.clone(),
        )
        .await
        .map_err(|e| Status::internal(format!("Failed to create sparse index: {}", e)))?;

        Ok(Response::new(()))
    }
}
}

#[cfg(test)]
mod tests {
    // Tests commented out until proper environment setup is available
}
