use actix_web::{web, Scope};

mod controller;
pub(crate) mod dtos;
mod error;
mod service;

pub(crate) fn version_module() -> Scope {
    let version_module = web::scope("/collections/{collection_id}/versions").
	route("", web::get().to(controller::list_versions))
	.route("/current", web::get().to(controller::get_current_version));
	// .route("/current", web::put().to(controller::set_current_version));
	version_module
}
