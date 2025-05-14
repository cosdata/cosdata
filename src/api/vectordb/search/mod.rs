use actix_web::{web, Scope};
use crate::rbac::guards::{
    require_query_dense_vectors, require_query_hybrid_vectors, require_query_sparse_vectors,
};

mod controller;
pub(crate) mod dtos;
pub(crate) mod error;
pub(crate) mod repo;
mod service;

pub(crate) fn search_module() -> Scope {
    web::scope("/search")
        .route(
            "/dense",
            web::post()
                .to(controller::dense_search)
                .wrap(require_query_dense_vectors()),
        )
        .route(
            "/batch-dense",
            web::post()
                .to(controller::batch_dense_search)
                .wrap(require_query_dense_vectors()),
        )
        .route(
            "/sparse",
            web::post()
                .to(controller::sparse_search)
                .wrap(require_query_sparse_vectors()),
        )
        .route(
            "/batch-sparse",
            web::post()
                .to(controller::batch_sparse_search)
                .wrap(require_query_sparse_vectors()),
        )
        .route(
            "/tf-idf",
            web::post()
                .to(controller::tf_idf_search)
                .wrap(require_query_sparse_vectors()),
        )
        .route(
            "/batch-tf-idf",
            web::post()
                .to(controller::batch_tf_idf_search)
                .wrap(require_query_sparse_vectors()),
        )
        .route(
            "/hybrid",
            web::post()
                .to(controller::hybrid_search)
                .wrap(require_query_hybrid_vectors()),
        )
}
