use actix_web::{web, Scope};
use crate::rbac::guards::{
    require_list_collections, require_create_collection, require_delete_collection,
    require_update_collection
};

mod controller;
mod dtos;
mod error;
mod repo;
pub(crate) mod service;

pub(crate) fn collections_module() -> Scope {
    web::scope("/collections")
        .route("", web::post().to(controller::create_collection)
               .wrap(require_create_collection()))
        .route("", web::get().to(controller::list_collections)
               .wrap(require_list_collections()))
        .route("/loaded", web::get().to(controller::get_loaded_collections)
               .wrap(require_list_collections()))
        .route("/{collection_id}", web::get().to(controller::get_collection_by_id)
               .wrap(require_list_collections()))
        .route(
            "/{collection_id}/indexing_status",
            web::get().to(controller::get_collection_indexing_status)
            .wrap(require_list_collections())
        )
        .route("/{collection_id}", web::delete().to(controller::delete_collection_by_id)
               .wrap(require_delete_collection()))
        .route("/{collection_id}/load", web::post().to(controller::load_collection)
               .wrap(require_update_collection()))
        .route("/{collection_id}/unload", web::post().to(controller::unload_collection)
               .wrap(require_update_collection()))
}
