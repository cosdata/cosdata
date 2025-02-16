use actix_web::{web, Scope};
use controller::{create_dense_index, create_sparse_index};

mod controller;
pub(crate) mod dtos;
mod error;
mod repo;
mod service;

pub(crate) fn indexes_module() -> Scope {
    let indexes_module = web::scope("/collections/{collection_id}/indexes")
        .route("/dense", web::post().to(create_dense_index))
        .route("/sparse", web::post().to(create_sparse_index));

    indexes_module
}
