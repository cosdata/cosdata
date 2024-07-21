mod api_service;
use crate::models::cache_loader::load_cache;
mod models;
mod vector_store;
mod web_server;
use web_server::run_actix_server;
pub(crate) mod api;

use crate::models::common::*;
use crate::models::lookup_table::*;

fn main() {
    // Initialize the lookup table once
    initialize_u16_lookup_table();

    let _ = run_actix_server();
    load_cache();
    ()
}
