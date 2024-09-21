mod commit;
mod controller;
mod delete;
mod dtos;
mod error;
mod repo;
mod service;
mod update;
mod upsert;

use actix_web::{web, Scope};
pub(crate) use commit::commit;
pub(crate) use delete::delete;
pub(crate) use update::update;
pub(crate) use upsert::upsert;

pub(crate) fn transactions_module() -> Scope {
    let transactions_module = web::scope("/collections/{collection_id}/transactions")
        .route("", web::post().to(controller::create_transaction))
        .route(
            "/{transaction_id}/abort",
            web::post().to(controller::abort_transaction),
        );
    transactions_module
}
