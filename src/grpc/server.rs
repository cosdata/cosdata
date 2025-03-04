use std::sync::Arc;
use tonic::transport::Server;
use log::info;

use crate::app_context::AppContext;
use super::collections::CollectionsServiceImpl;
use super::vectors::VectorsServiceImpl;
use super::auth::AuthServiceImpl;
use super::indexes::IndexesServiceImpl;
use super::proto::{
    collections_service_server::CollectionsServiceServer,
    vectors_service_server::VectorsServiceServer,
    auth_service_server::AuthServiceServer,
    indexes_service_server::IndexesServiceServer,
};
use tonic_reflection::server::{ServerReflection, ServerReflectionServer};

fn reflection_service() -> ServerReflectionServer<impl ServerReflection> {
    tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(super::proto::FILE_DESCRIPTOR_SET)
        .build()
        .unwrap()
}

pub async fn start_grpc_server(context: Arc<AppContext>, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    const DEFAULT_HOST: &str = "[::1]";
    let addr = format!("{}:{}", DEFAULT_HOST, port).parse()?;

    let collections_service = CollectionsServiceImpl {
        context: context.clone(),
    };
    let vectors_service = VectorsServiceImpl {
        context: context.clone(),
    };
    let auth_service = AuthServiceImpl {
        context: context.clone(),
    };
    let indexes_service = IndexesServiceImpl {
        context: context.clone(),
    };

    info!("gRPC server listening on {}", addr);
    Server::builder()
        .add_service(CollectionsServiceServer::new(collections_service))
        .add_service(VectorsServiceServer::new(vectors_service))
        .add_service(AuthServiceServer::new(auth_service))
        .add_service(IndexesServiceServer::new(indexes_service))
        .add_service(reflection_service())
        .serve(addr)
        .await?;

    Ok(())
}
