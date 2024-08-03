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

fn main() {
    let _ = run_actix_server();
    load_cache();
    ()
}
