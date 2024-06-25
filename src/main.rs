mod api_service;
mod models;
mod vector_store;
mod web_server;
use async_std::task;
use lazy_static::lazy_static;
use rand::Rng;
use std::f64;
use std::sync::{Arc, Mutex};
use web_server::run_actix_server;

use crate::models::common::*;
use crate::models::lookup_table::*;

fn main() {
    // Initialize the lookup table once
    initialize_u16_lookup_table();

    let _ = run_actix_server();
    ()
}
