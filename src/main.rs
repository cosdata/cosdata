mod api_service;
mod app_context;
pub mod macros;
mod models;
mod vector_store;
mod web_server;
use web_server::run_actix_server;
pub(crate) mod api;
pub mod config_loader;
pub mod cosql;
pub mod distance;
pub mod indexes;
pub mod quantization;
pub mod storage;
pub mod grpc;

use std::sync::Arc;
use crate::app_context::AppContext;

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = config_loader::load_config();
    // Create context
    let context = Arc::new(AppContext::new(config)?);

    // Start gRPC server
    let grpc_context = context.clone();
    let grpc_handle = actix_web::rt::spawn(async move {
        const DEFAULT_GRPC_PORT: u16 = 50051;
        if let Err(e) = grpc::server::start_grpc_server(grpc_context, DEFAULT_GRPC_PORT).await {
            log::error!("gRPC server error: {}", e);
        }
    });

    // Create a separate task for HTTP server
    let http_handle = actix_web::rt::spawn(async move {
        if let Err(e) = run_actix_server() {
            log::error!("HTTP server error: {}", e);
        }
    });

    // Wait for both servers
    let (grpc_result, http_result) = futures::join!(grpc_handle, http_handle);

    // Handle errors
    if let Err(e) = grpc_result {
        log::error!("gRPC server join error: {}", e);
    }
    if let Err(e) = http_result {
        log::error!("HTTP server join error: {}", e);
    }

    Ok(())
}
