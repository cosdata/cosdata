use crate::rbac::guards::{
    require_create_collection, require_delete_collection, require_list_collections,
    require_update_collection,
};
use actix_web::{web, Scope};

use crate::api::vectordb::indexes::indexes_module;
use crate::api::vectordb::search::search_module;
use crate::api::vectordb::sync_transaction::sync_transactions_module;
use crate::api::vectordb::transactions::transactions_module;
use crate::api::vectordb::vectors::vectors_module;
use crate::api::vectordb::versions::version_module;

mod controller;
mod dtos;
mod error;
mod repo;
pub(crate) mod service;

pub(crate) fn collections_module() -> Scope {
    web::scope("")
        .route(
            "/collections",
            web::post()
                .to(controller::create_collection)
                .wrap(require_create_collection()),
        )
        .route(
            "/collections",
            web::get()
                .to(controller::list_collections)
                .wrap(require_list_collections()),
        )
        .route(
            "/collections/loaded",
            web::get()
                .to(controller::get_loaded_collections)
                .wrap(require_list_collections()),
        )
        .service(
            web::scope("/collections/{collection_id}")
                .route(
                    "",
                    web::get()
                        .to(controller::get_collection_by_id)
                        .wrap(require_list_collections()),
                )
                .route(
                    "/indexing_status",
                    web::get()
                        .to(controller::get_collection_indexing_status)
                        .wrap(require_list_collections()),
                )
                .route(
                    "",
                    web::delete()
                        .to(controller::delete_collection_by_id)
                        .wrap(require_delete_collection()),
                )
                .route(
                    "/load",
                    web::post()
                        .to(controller::load_collection)
                        .wrap(require_update_collection()),
                )
                .route(
                    "/unload",
                    web::post()
                        .to(controller::unload_collection)
                        .wrap(require_update_collection()),
                )
                .service(indexes_module())
                .service(search_module())
                .service(vectors_module())
                .service(transactions_module())
                .service(sync_transactions_module())
                .service(version_module()),
        )
}
