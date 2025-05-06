use actix_web::{web, Scope};
use crate::rbac::guards::{
    require_list_index, require_create_index, require_delete_index
};

pub(crate) mod controller;
pub(crate) mod dtos;
mod error;
mod repo;
mod service;

pub(crate) fn indexes_module() -> Scope {
    web::scope("/collections/{collection_id}/indexes")
        .route("", web::get().to(controller::get_index)
               .wrap(require_list_index()))
        .route("/dense", web::post().to(controller::create_dense_index)
               .wrap(require_create_index()))
        .route("/sparse", web::post().to(controller::create_sparse_index)
               .wrap(require_create_index()))
        .route("/tf-idf", web::post().to(controller::create_tf_idf_index)
               .wrap(require_create_index()))
        .route("/{index_type}", web::delete().to(controller::delete_index)
               .wrap(require_delete_index()))
}
