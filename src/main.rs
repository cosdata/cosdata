mod api_service;
mod app_context;
pub mod macros;
mod models;
mod vector_store;
mod web_server;
pub(crate) mod api;
pub mod config_loader;
pub mod cosql;
pub mod distance;
pub mod indexes;
pub mod quantization;
pub mod storage;
pub mod grpc;

use std::sync::Arc;
use crate::{
    app_context::AppContext,
    web_server::run_actix_server_with_context,
};


#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let config = config_loader::load_config();
    // Create context
    let context = Arc::new(AppContext::new(config)?);

    // Start gRPC server
    let grpc_context = context.clone();
    actix_web::rt::spawn(async move {
        const DEFAULT_GRPC_PORT: u16 = 50051;
        if let Err(e) = grpc::server::start_grpc_server(grpc_context, DEFAULT_GRPC_PORT).await {
            log::error!("gRPC server error: {}", e);
        }
    });

    if let Err(e) = run_actix_server_with_context(context).await {
        log::error!("HTTP server error: {}", e);
    }

    Ok(())
}
