use crate::{models::rpc::*, vector_store::*};
use actix_cors::Cors;
use actix_files::Files;
use actix_web::{
    dev::ServiceRequest, http::header::ContentType, middleware, web, App, Error, HttpRequest,
    HttpResponse, HttpServer,
};
use actix_web_httpauth::{extractors::bearer::BearerAuth, middleware::HttpAuthentication};
use log::debug;
use rustls::{pki_types::PrivateKeyDer, ServerConfig};
use rustls_pemfile::{certs, pkcs8_private_keys};
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use std::{fs::File, io::BufReader};
use waco::models::rpc::Authenticate;
use waco::models::rpc::CreateVectorDb;

#[derive(Debug, Serialize, Deserialize)]
struct MyObj {
    name: String,
    number: i32,
}

async fn validator(
    req: ServiceRequest,
    credentials: BearerAuth,
) -> Result<ServiceRequest, (Error, ServiceRequest)> {
    println!("cred: {credentials:?}");
    Ok(req)
}

/// This handler uses json extractor
async fn authenticate(item: web::Json<Authenticate>) -> HttpResponse {
    println!("model: {:?}", &item);
    HttpResponse::Ok().json(item.0) // <- send response
}

/// This handler uses json extractor
async fn create_vector_db(item: web::Json<CreateVectorDb>) -> HttpResponse {
    // Extract values from the JSON request
    let vector_db_name = &item.vector_db_name;
    let dimensions = item.dimensions;
    let max_val = item.max_val;
    let min_val = item.min_val;

    // Define the parameters for init_vector_store
    let name = vector_db_name.clone();
    let size = dimensions as usize;
    let lower_bound = min_val;
    let upper_bound = max_val;
    let max_cache_level = 5;

    // Call init_vector_store using web::block
    let result = init_vector_store(name, size, lower_bound, upper_bound, max_cache_level).await;

    match result {
        Ok(__) => HttpResponse::Ok().json(RPCResponseBody::RespCreateVectorDb { result: true }),
        Err(e) => HttpResponse::NotAcceptable().body(format!("Error: {}", e)),
    }
}
/// This handler uses json extractor
async fn upsert_vector_db(item: web::Json<UpsertVectors>) -> HttpResponse {
    // Extract values from the JSON request
    let vector_db_name = &item.vector_db_name;
    let vector = item.vector.clone(); // Clone the vector for async usage

    let ain_env = get_app_env();

    // Try to get the vector store from the environment
    let vec_store = match ain_env.vector_store_map.get(vector_db_name) {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    // Call run_upload with the extracted parameters
    let __result = run_upload(vec_store.clone().into(), vector).await;

    // Placeholder response
    let response_data = RPCResponseBody::RespUpsertVectors { insert_stats: None }; //
    let response = HttpResponse::Ok().json(response_data);

    response // Return the response
}

/// This handler uses json extractor
async fn search_vector_db(item: web::Json<VectorANN>) -> HttpResponse {
    println!("model: {:?}", &item);
    HttpResponse::Ok().json(item.0) // <- send response
}

/// This handler uses json extractor with limit
async fn extract_item(item: web::Json<MyObj>, req: HttpRequest) -> HttpResponse {
    println!("request: {req:?}");
    println!("model: {item:?}");

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

    let config = load_rustls_config();

    log::info!("starting HTTPS server at https://localhost:8443");

    HttpServer::new(|| {
        let auth = HttpAuthentication::bearer(validator);
        App::new()
            // enable logger
            .wrap(middleware::Logger::default())
            // ensure the CORS middleware is wrapped around the httpauth middleware so it is able to
            // add headers to error responses
            .wrap(Cors::permissive())
            // register simple handler, handle all methods
            .app_data(web::JsonConfig::default().limit(4 * 1048576)) // <- 4 mb limit size of the payload (global configuration)
            .service(
                web::scope("/auth")
                    .service(web::resource("/gettoken").route(web::post().to(authenticate))),
            )
            .service(
                web::scope("/vectordb")
                    .wrap(auth.clone())
                    .service(web::resource("/createdb").route(web::post().to(create_vector_db)))
                    .service(web::resource("/upsert").route(web::post().to(upsert_vector_db)))
                    .service(web::resource("/search").route(web::post().to(search_vector_db))),
            )
        // .service(web::resource("/index").route(web::post().to(index)))
        // .service(
        //     web::resource("/extractor")
        //         .app_data(web::JsonConfig::default().limit(1024)) // <- limit size of the payload (resource level)
        //         .route(web::post().to(extract_item)),
        // )
        // .service(web::resource("/manual").route(web::post().to(index_manual)))
        // .service(web::resource("/").route(web::post().to(index)))
    })
    .bind_rustls_0_23("127.0.0.1:8443", config)?
    .run()
    .await
}

fn load_rustls_config() -> rustls::ServerConfig {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .unwrap();

    // init server config builder with safe defaults
    let config = ServerConfig::builder().with_no_client_auth();

    // load TLS key/cert files
    let cert_file = &mut BufReader::new(File::open("/home/nithin/example.crt").unwrap());
    let key_file = &mut BufReader::new(File::open("/home/nithin/example.key").unwrap());

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
