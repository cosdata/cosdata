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

use crate::models::common::*;

fn main() {
    let _ = run_actix_server();
}
