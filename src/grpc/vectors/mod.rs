// contains errors
// #[cfg(test)]
// mod tests;

use crate::models::common::WaCustomError;
use crate::{app_context::AppContext, indexes::inverted::types::SparsePair};
use std::sync::Arc;
use tonic::{Request, Response, Status};

crate::cfg_grpc! {
use super::proto::{
    GetVectorRequest,
    FindSimilarVectorsRequest, FindSimilarVectorsResponse,
    SimilarVectorMatch, VectorResponse,
    vectors_service_server::VectorsService,
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
        let _collection = self.context.ain_env.collections_map
            .get_collection(&req.collection_id)
            .ok_or_else(|| Status::not_found(format!("Collection '{}' not found", req.collection_id)))?;

        unimplemented!()
    }

    // Finds similar vectors based on a query vector
    async fn find_similar_vectors(
        &self,
        request: Request<FindSimilarVectorsRequest>,
    ) -> Result<Response<FindSimilarVectorsResponse>, Status> {
        let req = request.into_inner();

        // Validate collection exists
        let collection = self.context.ain_env.collections_map
            .get_collection(&req.collection_id)
            .ok_or_else(|| Status::not_found(format!("Collection '{}' not found", req.collection_id)))?;

        match req.query {
            // Handle dense vector similarity search
            Some(super::proto::find_similar_vectors_request::Query::Dense(dense)) => {
                if !collection.meta.dense_vector.enabled {
                    return Err(Status::failed_precondition("Dense vectors are not enabled for this collection"));
                }

                let hnsw_index = collection
                    .get_hnsw_index()
                    .ok_or_else(|| Status::failed_precondition("Dense index not initialized"))?;

                // Perform similarity search
                let results = crate::api_service::ann_vector_query(
                    self.context.clone(),
                    &collection,
                    hnsw_index,
                    dense.vector,
                    // @TODO: Support for metadata filtering to be
                    // added for grpc endpoints
                    None,
                    dense.top_k.map(|top_k| top_k as usize)
                ).await.map_err(|e| match e {
                    WaCustomError::NotFound(msg) => Status::not_found(msg),
                    _ => Status::internal(format!("Failed to find similar vectors: {}", e))
                })?;

                Ok(Response::new(FindSimilarVectorsResponse {
                    results: Some(super::proto::SearchResults {
                        matches: results.into_iter()
                            .map(|(id, score)| SimilarVectorMatch {
                                id: id.0,
                                score: score.get_value(),
                            })
                            .collect(),
                    })
                }))
            }

            // Sparse vector similarity search not implemented
            Some(super::proto::find_similar_vectors_request::Query::Sparse(sparse)) => {
                if !collection.meta.sparse_vector.enabled {
                    return Err(Status::failed_precondition("Sparse vectors are not enabled for this collection"));
                }

                let inverted_index = collection.get_inverted_index().ok_or_else(|| Status::failed_precondition("Sparse index not initialized"))?;

                let query: Vec<_> = sparse.values.into_iter().map(|pair| SparsePair(pair.index, pair.value)).collect();

                let results = crate::api::vectordb::search::repo::sparse_ann_vector_query_logic(&self.context.config, inverted_index, &query, sparse.top_k.map(|top_k| top_k as usize), sparse.early_terminate_threshold.unwrap_or(self.context.config.search.early_terminate_threshold)).map_err(Status::from)?;

                Ok(Response::new(FindSimilarVectorsResponse {
                    results: Some(super::proto::SearchResults {
                        matches: results.into_iter().map(|(id, score)| SimilarVectorMatch {
                            id: id.0,
                            score: score.get_value(),
                        }).collect(),
                    })
                }))
            }
            Some(super::proto::find_similar_vectors_request::Query::TfIdf(idf)) => {
                if !collection.meta.tf_idf_options.enabled {
                    return Err(Status::failed_precondition("TF-IDF index is not enabled for this collection"));
                }

                let inverted_index = collection.get_tf_idf_index().ok_or_else(|| Status::failed_precondition("Sparse index not initialized"))?;

                let results = crate::api::vectordb::search::repo::tf_idf_ann_vector_query(inverted_index, &idf.query, idf.top_k.map(|top_k| top_k as usize)).map_err(Status::from)?;

                Ok(Response::new(FindSimilarVectorsResponse {
                    results: Some(super::proto::SearchResults {
                        matches: results.into_iter().map(|(id, score)| SimilarVectorMatch {
                            id: id.0,
                            score,
                        }).collect(),
                    })
                }))

            }
            None => Err(Status::invalid_argument("Query must be specified")),
        }
    }
}
}
