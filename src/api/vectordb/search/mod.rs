use actix_web::{web, Scope};
use controller::{dense_search, batch_dense_search, sparse_search, batch_sparse_search, hybrid_search, sparse_idf_search, batch_sparse_idf_search};

mod controller;
pub(crate) mod dtos;
pub(crate) mod error;
mod repo;
mod service;

pub(crate) fn search_module() -> Scope {
    web::scope("/collections/{collection_id}/search")
        .route("/dense", web::post().to(dense_search))
        .route("/batch-dense", web::post().to(batch_dense_search))
        .route("/sparse", web::post().to(sparse_search))
        .route("/batch-sparse", web::post().to(batch_sparse_search))
        .route("/sparse/idf", web::post().to(sparse_idf_search))
        .route("/batch-sparse/idf", web::post().to(batch_sparse_idf_search))
        .route("/hybrid", web::post().to(hybrid_search))
}
