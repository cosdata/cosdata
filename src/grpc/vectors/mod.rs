use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::app_context::AppContext;
use crate::models::types::VectorId;
use crate::indexes::inverted_index_types::SparsePair;

use super::proto::{
    CreateVectorRequest, GetVectorRequest,
    UpdateVectorRequest, DeleteVectorRequest,
    FindSimilarVectorsRequest, FindSimilarVectorsResponse,
    SimilarVectorMatch, VectorResponse, DenseVector, SparseVector,
    vectors_service_server::VectorsService,
};

pub struct VectorsServiceImpl {
    pub context: Arc<AppContext>,
}

#[tonic::async_trait]
impl VectorsService for VectorsServiceImpl {
    async fn create_vector(
        &self,
        request: Request<CreateVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();

        match req.vector {
            Some(super::proto::create_vector_request::Vector::Dense(dense)) => {
                let vec_to_insert = vec![(VectorId(dense.id), dense.values.clone())];

                let dense_index = self.context.ain_env.collections_map
                    .get(&req.collection_id)
                    .ok_or_else(|| Status::not_found("Dense index not found"))?;

                crate::api_service::run_upload(
                    self.context.clone(),
                    dense_index,
                    vec_to_insert,
                ).map_err(Status::from)?;

                Ok(Response::new(VectorResponse {
                    vector: Some(super::proto::vector_response::Vector::Dense(DenseVector {
                        id: dense.id,
                        values: dense.values,
                    }))
                }))
            }

            Some(super::proto::create_vector_request::Vector::Sparse(sparse)) => {
                let sparse_pairs: Vec<SparsePair> = sparse.values.into_iter()
                    .map(|p| SparsePair(p.index, p.value))
                    .collect();

                let vec_to_insert = vec![(VectorId(sparse.id), sparse_pairs.clone())];

                let sparse_index = self.context.ain_env.collections_map
                    .get_inverted_index(&req.collection_id)
                    .ok_or_else(|| Status::not_found("Sparse index not found"))?;

                crate::api_service::run_upload_sparse_vector(
                    sparse_index,
                    vec_to_insert,
                ).map_err(Status::from)?;

                Ok(Response::new(VectorResponse {
                    vector: Some(super::proto::vector_response::Vector::Sparse(SparseVector {
                        id: sparse.id,
                        values: sparse_pairs.into_iter()
                            .map(|p| super::proto::SparsePair {
                                index: p.0,
                                value: p.1,
                            })
                            .collect(),
                    }))
                }))
            }

            None => Err(Status::invalid_argument("Vector must be specified")),
        }
    }

    async fn get_vector(
        &self,
        request: Request<GetVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();

        let dense_index = self.context.ain_env.collections_map
            .get(&req.collection_id)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let embedding = crate::vector_store::get_embedding_by_id(
            dense_index,
            &VectorId(req.vector_id)
        ).map_err(Status::from)?;

        Ok(Response::new(VectorResponse {
            vector: Some(super::proto::vector_response::Vector::Dense(DenseVector {
                id: embedding.hash_vec.0,
                values: (*embedding.raw_vec).clone(),
            }))
        }))
    }

    async fn update_vector(
        &self,
        request: Request<UpdateVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();

        let dense_index = self.context.ain_env.collections_map
            .get(&req.collection_id)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let vec_to_update = vec![(VectorId(req.vector_id), req.values.clone())];
        crate::api_service::run_upload(
            self.context.clone(),
            dense_index.clone(),
            vec_to_update,
        ).map_err(Status::from)?;

        Ok(Response::new(VectorResponse {
            vector: Some(super::proto::vector_response::Vector::Dense(DenseVector {
                id: req.vector_id,
                values: req.values,
            }))
        }))
    }

    async fn delete_vector(
        &self,
        _request: Request<DeleteVectorRequest>,
    ) -> Result<Response<()>, Status> {
        Err(Status::unimplemented("Delete operation is not implemented"))
    }

    async fn find_similar_vectors(
        &self,
        request: Request<FindSimilarVectorsRequest>,
    ) -> Result<Response<FindSimilarVectorsResponse>, Status> {
        let req = request.into_inner();

        match req.query {
            Some(super::proto::find_similar_vectors_request::Query::Dense(dense)) => {
                let dense_index = self.context.ain_env.collections_map
                    .get(&req.collection_id)
                    .ok_or_else(|| Status::not_found("Dense index not found"))?;

                let results = crate::api_service::ann_vector_query(
                    self.context.clone(),
                    dense_index,
                    dense.vector,
                    Some(dense.k as usize),
                ).await.map_err(Status::from)?;

                Ok(Response::new(FindSimilarVectorsResponse {
                    results: Some(super::proto::find_similar_vectors_response::Results::Dense(
                        super::proto::DenseResults {
                            matches: results.into_iter()
                                .map(|(id, score)| SimilarVectorMatch {
                                    id: id.0,
                                    score: score.get_value(),
                                })
                                .collect(),
                        }
                    ))
                }))
            }

            Some(super::proto::find_similar_vectors_request::Query::Sparse(_)) => {
                Err(Status::unimplemented("Sparse vector similarity search not yet implemented"))
            }

            None => Err(Status::invalid_argument("Query must be specified")),
        }
    }
}
