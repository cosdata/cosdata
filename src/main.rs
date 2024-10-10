mod api_service;
mod app_context;
use crate::models::cache_loader::load_cache;
mod models;
mod vector_store;
mod web_server;
use web_server::run_actix_server;
pub(crate) mod api;
pub mod config_loader;
pub mod cosql;
pub mod distance;
pub mod quantization;
pub mod storage;

use crate::models::common::*;

fn main() {
    load_cache();
    let _ = run_actix_server();
}
