use actix_web::web::Data;
use clap::Parser;

use cosdata::{app_context::AppContext, args::CosdataArgs, web_server::run_actix_server_with_context};

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CosdataArgs::parse();
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let config = cosdata::config_loader::load_config()?;
    // Create context
    let context = Data::new(AppContext::new(config, args)?);

    // Start gRPC server
    #[cfg(feature = "grpc-server")]
    let grpc_context = context.clone().into_inner();

    #[cfg(feature = "grpc-server")]
    actix_web::rt::spawn(async move {
        const DEFAULT_GRPC_PORT: u16 = 50051;
        if let Err(e) = cosdata::grpc::server::start_grpc_server(grpc_context, DEFAULT_GRPC_PORT).await {
            log::error!("gRPC server error: {}", e);
        }
    });

    if let Err(e) = run_actix_server_with_context(context).await {
        log::error!("HTTP server error: {}", e);
    }

    Ok(())
}
