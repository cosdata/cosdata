pub mod controller;
mod repo;
mod service;

use actix_web::{web, Scope};

pub(crate) fn streaming_module() -> Scope {
    web::scope("/collections/{collection_id}/streaming")
        .route("/upsert", web::post().to(controller::upsert))
        .route("/om_upsert", web::post().to(controller::om_upsert))
        .route(
            "/vectors/{vector_id}",
            web::delete().to(controller::delete_vector_by_id),
        )
}
