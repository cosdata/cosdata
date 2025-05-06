use actix_web::{web, Scope};
use crate::rbac::guards::{
    require_list_versions, require_get_current_version
};

mod controller;
pub(crate) mod dtos;
mod error;
mod service;

pub(crate) fn version_module() -> Scope {
    web::scope("/collections/{collection_id}/versions")
        .route("", web::get().to(controller::list_versions).wrap(require_list_versions()))
        .route("/current", web::get().to(controller::get_current_version).wrap(require_get_current_version()))
}
