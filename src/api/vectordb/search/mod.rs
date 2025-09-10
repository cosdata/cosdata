use actix_web::{web, Scope};
use controller::{
    batch_dense_search, batch_hybrid_search, batch_om_sum_query, batch_sparse_search,
    batch_tf_idf_search, dense_search, hybrid_search, om_sum_query, sparse_search, tf_idf_search,
};

pub mod controller;
pub(crate) mod dtos;
pub(crate) mod error;
pub(crate) mod repo;
mod service;

pub(crate) fn search_module() -> Scope {
    web::scope("/collections/{collection_id}/search")
        .route("/dense", web::post().to(dense_search))
        .route("/batch-dense", web::post().to(batch_dense_search))
        .route("/sparse", web::post().to(sparse_search))
        .route("/batch-sparse", web::post().to(batch_sparse_search))
        .route("/tf-idf", web::post().to(tf_idf_search))
        .route("/batch-tf-idf", web::post().to(batch_tf_idf_search))
        .route("/hybrid", web::post().to(hybrid_search))
        .route("/batch-hybrid", web::post().to(batch_hybrid_search))
        .route("/om-sum", web::post().to(om_sum_query))
        .route("/batch-om-sum", web::post().to(batch_om_sum_query))
}
