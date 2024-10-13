use actix_web::{web, Scope};

mod controller;
pub(crate) mod dtos;
mod error;
pub(crate) mod repo;
mod service;

pub(crate) fn vectors_module() -> Scope {
    let vectors_module = web::scope("/collections/{collection_id}/vectors")
        .route("", web::post().to(controller::create_vector))
        .route("/search", web::post().to(controller::find_similar_vectors))
        .route("/{vector_id}", web::get().to(controller::get_vector_by_id))
        .route(
            "/{vector_id}",
            web::put().to(controller::update_vector_by_id),
        )
        .route(
            "/{vector_id}",
            web::delete().to(controller::delete_vector_by_id),
        );

    vectors_module
}
