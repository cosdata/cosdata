use actix_web::{web, Scope};

pub mod controller;
pub(crate) mod dtos;
mod error;
mod service;

pub(crate) fn version_module() -> Scope {
    web::scope("/collections/{collection_id}/versions")
        .route("", web::get().to(controller::list_versions))
        .route("/current", web::get().to(controller::get_current_version))
    // .route("/current", web::put().to(controller::set_current_version))
}
