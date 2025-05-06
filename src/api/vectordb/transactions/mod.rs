use actix_web::{web, Scope};
use crate::rbac::guards::{
    require_upsert_vectors, require_delete_vectors,
    require_list_vectors
};

mod controller;
pub(crate) mod delete;
mod dtos;
mod error;
mod repo;
mod service;
pub(crate) mod update;

pub(crate) fn transactions_module() -> Scope {
    web::scope("/collections/{collection_id}/transactions")
        .route("", web::post().to(controller::create_transaction)
               .wrap(require_upsert_vectors()))
        .route("/{transaction_id}/commit", web::post().to(controller::commit_transaction)
               .wrap(require_upsert_vectors()))
        .route(
            "/{transaction_id}/status",
            web::get().to(controller::get_transaction_status)
            .wrap(require_list_vectors())
        )
        .route("/{transaction_id}/vectors", web::post().to(controller::create_vector_in_transaction)
               .wrap(require_upsert_vectors()))
        .route("/{transaction_id}/upsert", web::post().to(controller::upsert)
               .wrap(require_upsert_vectors()))
        .route("/{transaction_id}/vectors/{vector_id}", web::delete().to(controller::delete_vector_by_id)
               .wrap(require_delete_vectors()))
        .route("/{transaction_id}/abort", web::post().to(controller::abort_transaction)
               .wrap(require_upsert_vectors()))
}
