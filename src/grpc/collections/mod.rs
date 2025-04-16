use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::app_context::AppContext;
use crate::metadata::schema::MetadataSchema;
use crate::models::collection::{
    Collection, CollectionConfig, DenseVectorOptions, SparseVectorOptions, TFIDFOptions,
};

crate::cfg_grpc! {
use super::proto::collections_service_server::CollectionsService;
use super::proto::{
    Collection as ProtoCollection,
    CreateCollectionRequest, CreateCollectionResponse,
    GetCollectionsRequest, GetCollectionsResponse,
    GetCollectionRequest, DeleteCollectionRequest,
};

pub struct CollectionsServiceImpl {
    pub context: Arc<AppContext>,
}

#[tonic::async_trait]
impl CollectionsService for CollectionsServiceImpl {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        // Create options from request
        let dense_vector = DenseVectorOptions {
            dimension: req.dense_vector.as_ref().map_or(0, |d| d.dimension as usize),
            enabled: req.dense_vector.as_ref().is_some_and(|d| d.enabled),
        };

        let sparse_vector = SparseVectorOptions {
            enabled: req.sparse_vector.as_ref().is_some_and(|d| d.enabled),
        };

        let tf_idf_options = TFIDFOptions {
            enabled: req.tf_idf_options.as_ref().is_some_and(|d| d.enabled),
        };

        let config = CollectionConfig {
            max_vectors: req.config.as_ref().and_then(|c| c.max_vectors.map(|v| v as i32)),
            replication_factor: req.config.as_ref().and_then(|c| c.replication_factor.map(|v| v as i32)),
        };

        // Convert metadata schema if present
        let metadata_schema: Option<MetadataSchema> = req.metadata_schema
            .map(|schema| schema.try_into())
            .transpose()
            .map_err(|e| Status::invalid_argument(format!("Invalid metadata schema: {}", e)))?;

        // Create new collection
        let collection = Collection::new(
            req.name.clone(),
            req.description.clone(),
            dense_vector,
            sparse_vector,
            tf_idf_options,
            metadata_schema,
            config,
        ).map_err(Status::from)?;

        // Store collection
        self.context.ain_env.collections_map
            .insert_collection(Arc::new(collection.clone()))
            .map_err(Status::from)?;

        Ok(Response::new(CreateCollectionResponse {
            id: collection.name.clone(),
            name: collection.name,
            description: collection.description,
        }))
    }

    async fn get_collections(
        &self,
        _request: Request<GetCollectionsRequest>,
    ) -> Result<Response<GetCollectionsResponse>, Status> {
        let collections = self.context.ain_env.collections_map
            .iter_collections()
            .map(|entry| ProtoCollection {
                name: entry.key().clone(),
                description: entry.value().description.clone(),
            })
            .collect();

        Ok(Response::new(GetCollectionsResponse { collections }))
    }

    async fn get_collection(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<ProtoCollection>, Status> {
        let collection = self.context.ain_env.collections_map
            .get_collection(&request.into_inner().id)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        Ok(Response::new(ProtoCollection {
            name: collection.name.clone(),
            description: collection.description.clone(),
        }))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<()>, Status> {
        let collection_id = request.into_inner().id;

        self.context.ain_env.collections_map
            .remove_collection(&collection_id)
            .map_err(|e| match e {
                crate::models::common::WaCustomError::NotFound(_) => Status::not_found("Collection not found"),
                _ => Status::internal(format!("Failed to delete collection: {}", e))
            })?;

        Ok(Response::new(()))
    }
}
}
