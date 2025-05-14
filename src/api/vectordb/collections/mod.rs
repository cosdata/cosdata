use actix_web::{web, Scope};
mod controller;
mod dtos;
mod error;
mod repo;
pub(crate) mod service;

pub(crate) fn collections_module() -> Scope {
    web::scope("/collections")
        .route("", web::post().to(controller::create_collection))
        .route("", web::get().to(controller::list_collections))
        .route("", web::get().to(controller::get_collections))
        .route("/loaded", web::get().to(controller::get_loaded_collections))
        .route(
            "/{collection_id}",
            web::get().to(controller::get_collection_by_id),
        )
        .route(
            "/{collection_id}/indexing_status",
            web::get().to(controller::get_collection_indexing_status),
        )
        .route(
            "/{collection_id}",
            web::delete().to(controller::delete_collection_by_id),
        )
        .route(
            "/{collection_id}/load",
            web::post().to(controller::load_collection),
        )
        .route(
            "/{collection_id}/unload",
            web::post().to(controller::unload_collection),
        )
}
