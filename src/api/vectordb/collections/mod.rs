use actix_web::{web, Scope};
mod controller;
mod dtos;
mod error;
mod repo;
pub(crate) mod service;

pub(crate) fn collections_module() -> Scope {
    let collections_module = web::scope("/collections")
        .route("", web::post().to(controller::create_collection))
        .route(
            "/{collection_id}",
            web::get().to(controller::get_collection_by_id),
        );

    collections_module
}
