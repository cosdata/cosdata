use actix_cors::Cors;
use actix_web::{
    dev::ServiceRequest, middleware, web, App, Error, HttpRequest, HttpResponse, HttpServer,
};
use actix_web_httpauth::{extractors::bearer::BearerAuth, middleware::HttpAuthentication};
use dashmap::DashMap;
use lmdb::Environment;
use rustls::{pki_types::PrivateKeyDer, ServerConfig};
use rustls_pemfile::{certs, pkcs8_private_keys};
use serde::{Deserialize, Serialize};
use std::fs::create_dir_all;
use std::path::Path;
use std::sync::Arc;
use std::{fs::File, io::BufReader};
use cosdata::config_loader::{load_config, ServerMode, Ssl};
use actix_web::web::Data;

use crate::models::types::*;
use crate::{api, WaCustomError};
use std::env;

#[derive(Debug, Serialize, Deserialize)]
struct MyObj {
    name: String,
    number: i32,
}

async fn validator(
    req: ServiceRequest,
    credentials: BearerAuth,
) -> Result<ServiceRequest, (Error, ServiceRequest)> {
    // println!("cred: {credentials:?}");
    Ok(req)
}

async fn extract_item(item: web::Json<MyObj>, req: HttpRequest) -> HttpResponse {
    // println!("request: {req:?}");
    // println!("model: {item:?}");

    HttpResponse::Ok().json(item.0) // <- send json response
}

/// This handler manually load request payload and parse json object
async fn index_manual(body: web::Bytes) -> Result<HttpResponse, Error> {
    // body is loaded, now we can deserialize serde-json
    let obj = serde_json::from_slice::<MyObj>(&body)?;
    Ok(HttpResponse::Ok().json(obj)) // <- send response
}

#[actix_web::main]
pub async fn run_actix_server() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let config = load_config();

    let tls = match &config.server.mode {
        ServerMode::Https => Some(load_rustls_config(&config.server.ssl)),
        ServerMode::Http => {
            log::warn!("server.mode=http is not recommended in production");
            None
        },
    };

    let config_data = Data::new(config);
    let app_state = config_data.clone();
    log::info!(
        "starting HTTPS server at {}://{}:{}",
        &config_data.server.mode.protocol(),
        &config_data.server.host,
        &config_data.server.port
    );

    let server = HttpServer::new(move || {
        let auth = HttpAuthentication::bearer(validator);
        App::new()
            // enable logger
            .wrap(middleware::Logger::default())
            // ensure the CORS middleware is wrapped around the httpauth middleware
            // so it is able to add headers to error responses
            .wrap(Cors::permissive())
            // register simple handler, handle all methods
            .app_data(web::JsonConfig::default().limit(4 * 1048576))
            .route(
                "/auth/gettoken",
                web::post().to(crate::api::auth::get_token),
            )
            .service(
                web::scope("/vectordb")
                    .wrap(auth.clone())
                    .service(
                        web::resource("/createdb").route(web::post().to(api::vectordb::create)),
                    )
                    .service(web::resource("/upsert").route(web::post().to(api::vectordb::upsert)))
                    .service(web::resource("/search").route(web::post().to(api::vectordb::search)))
                    .service(web::resource("/fetch").route(web::post().to(api::vectordb::fetch)))
                    .service(
                        web::scope("{database_name}/transactions")
                            .route("/", web::post().to(api::vectordb::transactions::create))
                            .route(
                                "/{transaction_id}/upsert",
                                web::post().to(api::vectordb::transactions::upsert),
                            )
                            .route(
                                "/{transaction_id}/update",
                                web::post().to(api::vectordb::transactions::update),
                            )
                            .route(
                                "/{transaction_id}/delete",
                                web::post().to(api::vectordb::transactions::delete),
                            )
                            .route(
                                "/{transaction_id}/commit",
                                web::post().to(api::vectordb::transactions::commit),
                            )
                            .route(
                                "/{transaction_id}/abort",
                                web::post().to(api::vectordb::transactions::abort),
                            ),
                    ),
            )
            .app_data(app_state.clone())

        // .service(web::resource("/index").route(web::post().to(index)))
        // .service(
        //     web::resource("/extractor")
        //         .app_data(web::JsonConfig::default().limit(1024))
        // <- limit size of the payload (resource level)
        //         .route(web::post().to(extract_item)),
        // )
        // .service(web::resource("/manual").route(web::post().to(index_manual)))
        // .service(web::resource("/").route(web::post().to(index)))
    });

    let addr = (config_data.server.host.as_str(), config_data.server.port);
    let server = match tls {
        Some(config) => server.bind_rustls_0_23(addr, config),
        None => server.bind(addr)
    };
    server?
        .run()
        .await
}

fn load_rustls_config(ssl_config: &Ssl) -> rustls::ServerConfig {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .unwrap();

    // init server config builder with safe defaults
    let mut config = ServerConfig::builder().with_no_client_auth();

    // load TLS key/cert files
    let cert_file = &mut BufReader::new(File::open(&ssl_config.cert_file).unwrap_or_else(|_| {
        eprintln!("Failed to open certificate file: {}", ssl_config.key_file.display());
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

fn old_load_rustls_config() -> rustls::ServerConfig {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .unwrap();

    // init server config builder with safe defaults
    let config = ServerConfig::builder().with_no_client_auth();

    let key = "SSL_CERT_DIR";
    match env::var_os(key) {
        Some(val) => println!("{key}: {val:?}"),
        None => println!("{key} is not defined in the environment."),
    }
    // load TLS key/cert files
    let cert_file = &mut BufReader::new(File::open("~/example.crt").unwrap());
    let key_file = &mut BufReader::new(File::open("~/example.key").unwrap());

    // convert files to key/cert objects
    let cert_chain = certs(cert_file).collect::<Result<Vec<_>, _>>().unwrap();
    let mut keys = pkcs8_private_keys(key_file)
        .map(|key| key.map(PrivateKeyDer::Pkcs8))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // exit if no keys could be parsed
    if keys.is_empty() {
        eprintln!("Could not locate PKCS 8 private keys.");
        std::process::exit(1);
    }

    config.with_single_cert(cert_chain, keys.remove(0)).unwrap()
}
