use actix_web::{web, Scope};
mod controller;
mod dtos;
mod error;
mod service;
mod repo;

pub(crate) fn collections_module() -> Scope {
    let collections_module = web::scope("/collections").route(
        "/{collection_id}",
        web::get().to(controller::get_collection_by_id),
    );

    collections_module
}
