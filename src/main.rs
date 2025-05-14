use actix_web::web::Data;
use args::{CosdataArgs, Command};
use clap::Parser;

use cosdata::app_context::AppContext;
use cosdata::args::CosdataArgs;
use cosdata::config_loader;
use cosdata::web_server::run_actix_server_with_context;

#[cfg(feature = "grpc-server")]
use cosdata::grpc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CosdataArgs::parse();
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let config = config_loader::load_config()?;

     // Clone command before creating context
    let command = args.command.clone();
    let context = Data::new(AppContext::new(config, args)?);
    
    match command {
        Some(Command::ResetPassword) => {
            println!("Admin password reset complete!");
            Ok(())
        }
        None => {
            // Start gRPC server
            #[cfg(feature = "grpc-server")]
            let grpc_context = context.clone().into_inner();

            #[cfg(feature = "grpc-server")]
            actix_web::rt::spawn(async move {
                const DEFAULT_GRPC_PORT: u16 = 50051;
                if let Err(e) = grpc::server::start_grpc_server(grpc_context, DEFAULT_GRPC_PORT).await {
                    log::error!("gRPC server error: {}", e);
                }
            });

            actix_web::rt::System::new().block_on(async {
                if let Err(e) = run_actix_server_with_context(context).await {
                    log::error!("HTTP server error: {}", e);
                }
            });
            Ok(())
        }
    }
}
