use actix_web::{web, Scope};

mod controller;
pub(crate) mod dtos;
pub(crate) mod error;
pub(crate) mod repo;
mod service;

pub(crate) fn vectors_module() -> Scope {
    web::scope("/collections/{collection_id}/vectors")
        .route("", web::post().to(controller::create_vector))
        .route("/upsert", web::post().to(controller::upsert_vectors))
        .route("/{vector_id}", web::get().to(controller::get_vector_by_id))
        .route(
            "/{vector_id}",
            web::put().to(controller::update_vector_by_id),
        )
        .route(
            "/{vector_id}",
            web::delete().to(controller::delete_vector_by_id),
        )
        .route("/{vector_id}", web::head().to(controller::check_vector_existence))
        .route("/{vector_id}/neighbors", web::get().to(controller::fetch_vector_neighbors))
        // .route("/search", web::post().to(controller::find_similar_vectors)) // Moved to search module
        // .route("/batch-search", web::post().to(controller::batch_search)) // Moved to search module
}
