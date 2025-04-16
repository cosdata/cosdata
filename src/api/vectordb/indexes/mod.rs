use actix_web::{web, Scope};
use controller::{create_dense_index, create_sparse_index, create_tf_idf_index, delete_index};

pub(crate) mod controller;
pub(crate) mod dtos;
mod error;
mod repo;
mod service;

pub(crate) fn indexes_module() -> Scope {
    web::scope("/collections/{collection_id}/indexes")
        .route("", web::get().to(controller::get_index))
        .route("/dense", web::post().to(create_dense_index))
        .route("/sparse", web::post().to(create_sparse_index))
        .route("/tf-idf", web::post().to(create_tf_idf_index))
        .route("/{index_type}", web::delete().to(delete_index))
}
