use serde::Deserialize;
use std::fs;
mod api_service;
use crate::models::cache_loader::load_cache;
mod models;
mod vector_store;
mod web_server;
use web_server::run_actix_server;
pub(crate) mod api;
pub mod distance;
pub mod quantization;
pub mod storage;

use crate::models::common::*;


#[derive(Deserialize)]
struct Config {
   server: Server,
}

#[derive(Deserialize)]
struct Server {
   host: String,
   port: String
}




fn main() {

    let config_contents = fs::read_to_string("config.toml").expect("Failed to config file");
    let config: Config = toml::from_str(&config_contents).expect("Failed to parse config file contents!");

    let _ = run_actix_server(&config.server.host, &config.server.port);
    load_cache();
    ()
}
