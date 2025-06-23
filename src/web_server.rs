use crate::api::auth::{auth_module, authentication_middleware::AuthenticationMiddleware};
use crate::api::docs::api_docs_module;
use crate::api::vectordb::collections::collections_module;
use crate::api::vectordb::indexes::indexes_module;
use crate::api::vectordb::search::search_module;
use crate::api::vectordb::streaming::streaming_module;
use crate::api::vectordb::transactions::transactions_module;
use crate::api::vectordb::vectors::vectors_module;
use crate::api::vectordb::versions::version_module;
use crate::app_context::AppContext;
use crate::config_loader::{ServerMode, Ssl};
use actix_cors::Cors;
use actix_web::web::Data;
use actix_web::{middleware, web, App, HttpServer};
use rustls::{pki_types::PrivateKeyDer, ServerConfig};
use rustls_pemfile::{certs, pkcs8_private_keys};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufReader};

#[derive(Debug, Serialize, Deserialize)]
struct MyObj {
    name: String,
    number: i32,
}

pub async fn run_actix_server_with_context(ctx: Data<AppContext>) -> std::io::Result<()> {
    let config = &ctx.config.clone();

    let tls = match &config.server.mode {
        ServerMode::Https => Some(load_rustls_config(&config.server.ssl)),
        ServerMode::Http => {
            log::warn!("server.mode=http is not recommended in production");
            None
        }
    };

    log::info!(
        "starting {} server at {}://{}:{}",
        &config.server.mode.protocol().to_uppercase(),
        &config.server.mode.protocol(),
        &config.server.host,
        &config.server.port,
    );

    let server = HttpServer::new(move || {
        App::new()
            // enable logger
            .wrap(middleware::Logger::default())
            // ensure the CORS middleware is wrapped around the httpauth middleware
            // so it is able to add headers to error responses
            .wrap(Cors::permissive())
            // register simple handler, handle all methods
            .app_data(web::JsonConfig::default().limit(8_388_608)) // 8 MB)
            .app_data(ctx.clone())
            .service(api_docs_module())
            .service(auth_module())
            .service(
                web::scope("/vectordb")
                    .wrap(AuthenticationMiddleware(
                        ctx.ain_env.active_sessions.clone(),
                    ))
                    // vectors module must be registered before collections module
                    // as its scope path is more specific than collections module
                    .service(search_module())
                    .service(indexes_module())
                    .service(vectors_module())
                    .service(transactions_module())
                    .service(streaming_module())
                    .service(version_module())
                    .service(collections_module()),
            )
    })
    .keep_alive(std::time::Duration::from_secs(10));

    let addr = config.server.listen_address();
    let server = match tls {
        Some(tls_config) => server.bind_rustls_0_23(addr, tls_config),
        None => server.bind(addr),
    };
    server?.run().await
}

fn load_rustls_config(ssl_config: &Ssl) -> rustls::ServerConfig {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .unwrap();

    // init server config builder with safe defaults
    let config = ServerConfig::builder().with_no_client_auth();

    // load TLS key/cert files
    let cert_file = &mut BufReader::new(File::open(&ssl_config.cert_file).unwrap_or_else(|_| {
        eprintln!(
            "Failed to open certificate file: {}",
            ssl_config.key_file.display()
        );
        std::process::exit(1);
    }));
    let key_file = &mut BufReader::new(File::open(&ssl_config.key_file).unwrap_or_else(|_| {
        eprintln!("Failed to open key file: {}", ssl_config.key_file.display());
        std::process::exit(1);
    }));

    // convert files to key/cert objects
    let cert_chain = certs(cert_file)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|_| {
            eprintln!("Failed to parse certificate chain.");
            std::process::exit(1);
        });
    let mut keys = pkcs8_private_keys(key_file)
        .map(|key| key.map(PrivateKeyDer::Pkcs8))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|_| {
            eprintln!("Failed to parse private keys.");
            std::process::exit(1);
        });

    // exit if no keys could be parsed
    if keys.is_empty() {
        eprintln!("Could not locate PKCS 8 private keys.");
        std::process::exit(1);
    }

    config.with_single_cert(cert_chain, keys.remove(0)).unwrap()
}
