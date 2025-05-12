use crate::rbac::guards::{require_delete_vectors, require_upsert_vectors};

mod controller;
mod repo;
mod service;

use actix_web::{web, Scope};

pub(crate) fn sync_transactions_module() -> Scope {
    web::scope("/sync-transaction")
        .route(
            "/upsert",
            web::post()
                .to(controller::upsert)
                .wrap(require_upsert_vectors()),
        )
        .route(
            "/vectors/{vector_id}",
            web::delete()
                .to(controller::delete_vector_by_id)
                .wrap(require_delete_vectors()),
        )
}
