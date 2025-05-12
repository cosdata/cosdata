mod controller;
mod repo;
mod service;

use actix_web::{web, Scope};

pub(crate) fn sync_transactions_module() -> Scope {
    web::scope("/collections/{collection_id}/sync_transaction")
        .route("/upsert", web::post().to(controller::upsert))
        .route(
            "/vectors/{vector_id}",
            web::delete().to(controller::delete_vector_by_id),
        )
}
