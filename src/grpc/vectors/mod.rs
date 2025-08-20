// contains errors
// #[cfg(test)]
// mod tests;

use crate::app_context::AppContext;
use crate::indexes::hnsw::{DenseSearchInput, DenseSearchOptions};
use crate::indexes::inverted::{SparseSearchInput, SparseSearchOptions, ZoneId};
use crate::indexes::tf_idf::{TFIDFSearchInput, TFIDFSearchOptions};
use crate::indexes::IndexOps;
use crate::models::common::WaCustomError;
use crate::models::types::VectorId;
use std::sync::Arc;
use tonic::{Request, Response, Status};

use super::proto::GeoFenceField;
// crate::cfg_grpc! {
use super::proto::{
    vectors_service_server::VectorsService, FindSimilarVectorsRequest, FindSimilarVectorsResponse,
    GetVectorRequest, SimilarVectorMatch, Vector, VectorResponse,
};

pub struct VectorsServiceImpl {
    pub context: Arc<AppContext>,
}

#[tonic::async_trait]
impl VectorsService for VectorsServiceImpl {
    // Retrieves a vector by ID from a collection
    async fn get_vector(
        &self,
        request: Request<GetVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();

        // Validate collection and vector type
        let collection = self
            .context
            .ain_env
            .collections_map
            .get_collection(&req.collection_id)
            .ok_or_else(|| {
                Status::not_found(format!("Collection '{}' not found", req.collection_id))
            })?;

        let internal_id = collection
            .external_to_internal_map
            .get_latest(&VectorId::from(req.vector_id))
            .ok_or_else(|| Status::not_found("Vector not found"))?;
        let vector = collection
            .get_raw_emb_by_internal_id(internal_id)
            .ok_or_else(|| Status::not_found("Vector not found"))?
            .clone();

        Ok(Response::new(VectorResponse {
            vector: Some(Vector {
                id: vector.id.into(),
                document_id: vector.document_id.map(Into::into),
                dense_values: vector.dense_values.unwrap_or_default(),
                geo_fence_values: vector
                    .geo_fence_values
                    .map(|values| {
                        values
                            .into_iter()
                            .map(|(name, value)| GeoFenceField { name, value })
                            .collect()
                    })
                    .unwrap_or_default(),
                text: vector.text,
            }),
        }))
    }

    // Finds similar vectors based on a query vector
    async fn find_similar_vectors(
        &self,
        request: Request<FindSimilarVectorsRequest>,
    ) -> Result<Response<FindSimilarVectorsResponse>, Status> {
        let req = request.into_inner();

        // Validate collection exists
        let collection = self
            .context
            .ain_env
            .collections_map
            .get_collection(&req.collection_id)
            .ok_or_else(|| {
                Status::not_found(format!("Collection '{}' not found", req.collection_id))
            })?;

        let warning = collection.is_indexing().then(|| {
            "Embeddings are currently being indexed; some results may be temporarily unavailable."
                .to_string()
        });

        match req.query {
            // Handle dense vector similarity search
            Some(super::proto::find_similar_vectors_request::Query::Dense(dense)) => {
                if !collection.meta.dense_vector.enabled {
                    return Err(Status::failed_precondition(
                        "Dense vectors are not enabled for this collection",
                    ));
                }

                let hnsw_index = collection
                    .get_hnsw_index()
                    .ok_or_else(|| Status::failed_precondition("Dense index not initialized"))?;

                // Perform similarity search
                let results = hnsw_index
                    .search(
                        &collection,
                        DenseSearchInput(
                            dense.vector,
                            // @TODO: Support for metadata filtering to be
                            // added for grpc endpoints
                            None,
                        ),
                        &DenseSearchOptions {
                            top_k: dense.top_k.map(|top_k| top_k as usize),
                        },
                        &self.context.config,
                        dense.return_raw_text.unwrap_or_default(),
                    )
                    .map_err(|e| match e {
                        WaCustomError::NotFound(msg) => Status::not_found(msg),
                        _ => Status::internal(format!("Failed to find similar vectors: {}", e)),
                    })?;

                Ok(Response::new(FindSimilarVectorsResponse {
                    results: Some(super::proto::SearchResults {
                        matches: results
                            .into_iter()
                            .map(
                                |(id, document_id, score, text, _matches)| SimilarVectorMatch {
                                    id: id.into(),
                                    document_id: document_id.map(Into::into),
                                    score,
                                    text,
                                },
                            )
                            .collect(),
                    }),
                    warning,
                }))
            }

            // Sparse vector similarity search not implemented
            Some(super::proto::find_similar_vectors_request::Query::GeoFence(geo_fence)) => {
                if !collection.meta.sparse_vector.enabled {
                    return Err(Status::failed_precondition(
                        "Sparse vectors are not enabled for this collection",
                    ));
                }

                let inverted_index = collection
                    .get_inverted_index()
                    .ok_or_else(|| Status::failed_precondition("Sparse index not initialized"))?;

                let results = inverted_index
                    .search(
                        &collection,
                        SparseSearchInput(geo_fence.query),
                        &SparseSearchOptions {
                            top_k: geo_fence.top_k.map(|top_k| top_k as usize),
                            early_terminate_threshold: geo_fence.early_terminate_threshold,
                            sort_by_distance: geo_fence.sort_by_distance,
                            coordinates: (
                                geo_fence.coordinates.unwrap_or_default().lat,
                                geo_fence.coordinates.unwrap_or_default().lon,
                            ),
                            zones: geo_fence.zones.into_iter().map(ZoneId::from).collect(),
                        },
                        &self.context.config,
                        geo_fence.return_raw_text.unwrap_or_default(),
                    )
                    .map_err(Status::from)?;

                Ok(Response::new(FindSimilarVectorsResponse {
                    results: Some(super::proto::SearchResults {
                        matches: results
                            .into_iter()
                            .map(
                                |(id, document_id, score, text, _matches)| SimilarVectorMatch {
                                    id: id.into(),
                                    document_id: document_id.map(Into::into),
                                    score,
                                    text,
                                },
                            )
                            .collect(),
                    }),
                    warning,
                }))
            }
            Some(super::proto::find_similar_vectors_request::Query::TfIdf(idf)) => {
                if !collection.meta.tf_idf_options.enabled {
                    return Err(Status::failed_precondition(
                        "TF-IDF index is not enabled for this collection",
                    ));
                }

                let tf_idf_index = collection
                    .get_tf_idf_index()
                    .ok_or_else(|| Status::failed_precondition("Sparse index not initialized"))?;

                let results = tf_idf_index
                    .search(
                        &collection,
                        TFIDFSearchInput(idf.query),
                        &TFIDFSearchOptions {
                            top_k: idf.top_k.map(|top_k| top_k as usize),
                        },
                        &self.context.config,
                        idf.return_raw_text.unwrap_or_default(),
                    )
                    .map_err(Status::from)?;

                Ok(Response::new(FindSimilarVectorsResponse {
                    results: Some(super::proto::SearchResults {
                        matches: results
                            .into_iter()
                            .map(
                                |(id, document_id, score, text, _matches)| SimilarVectorMatch {
                                    id: id.into(),
                                    document_id: document_id.map(Into::into),
                                    score,
                                    text,
                                },
                            )
                            .collect(),
                    }),
                    warning,
                }))
            }
            None => Err(Status::invalid_argument("Query must be specified")),
        }
    }
}
// }
