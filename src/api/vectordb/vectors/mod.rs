use actix_web::{web, Scope};

pub mod controller;
pub(crate) mod dtos;
pub(crate) mod error;
pub(crate) mod repo;
mod service;

pub(crate) fn vectors_module() -> Scope {
    web::scope("/collections/{collection_id}/vectors")
        .route("", web::get().to(controller::query_vectors))
        .route("/{vector_id}", web::get().to(controller::get_vector_by_id))
        .route(
            "/{vector_id}",
            web::head().to(controller::check_vector_existence),
        )
        .route(
            "/{vector_id}/neighbors",
            web::get().to(controller::fetch_vector_neighbors),
        )
}
