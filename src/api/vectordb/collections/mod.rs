use actix_web::{web, Scope};
mod controller;
mod dtos;
mod error;
mod repo;
pub(crate) mod service;

// Re-export the controller functions needed for routing
pub(crate) use controller::{
    create_collection,
    get_collections,
    get_collection_by_id,
    delete_collection_by_id,
    load_collection,
    unload_collection,
    get_loaded_collections,
};

pub(crate) fn collections_module() -> Scope {
    let collections_module = web::scope("/collections")
        .route("", web::post().to(controller::create_collection))
        .route("", web::get().to(controller::get_collections))
        .route(
            "/{collection_id}",
            web::get().to(controller::get_collection_by_id),
        )
        .route(
            "/{collection_id}",
            web::delete().to(controller::delete_collection_by_id),
        )
        // Collection cache management routes
        .route(
            "/{collection_id}/load",
            web::post().to(controller::load_collection),
        )
        .route(
            "/{collection_id}/unload",
            web::post().to(controller::unload_collection),
        )
        .route(
            "/loaded",
            web::get().to(controller::get_loaded_collections),
        );

    collections_module
}
