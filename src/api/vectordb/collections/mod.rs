use actix_web::{web, Scope, HttpResponse, Result};
mod controller;
mod dtos;
mod error;
mod repo;
pub(crate) mod service;

pub(crate) fn collections_module() -> Scope {
    web::scope("/collections")
        .route("", web::post().to(controller::create_collection))
        .route("", web::get().to(controller::get_collections))
        .route(
            "/{collection_id}",
            web::get().to(controller::get_collection_by_id),
        )
        .route(
            "/{collection_id}",
            web::delete().to(controller::delete_collection_by_id),
        )
        .route(
            "/{collection_id}/load",
            web::post().to(load_collection),
        )
        .route(
            "/{collection_id}/unload",
            web::post().to(unload_collection),
        )
        .route(
            "/loaded",
            web::get().to(get_loaded_collections),
        )
}

pub(crate) async fn load_collection(
    collection_id: web::Path<String>,
    ctx: web::Data<crate::app_context::AppContext>,
) -> Result<HttpResponse> {
    controller::load_collection(collection_id, ctx).await
}

pub(crate) async fn unload_collection(
    collection_id: web::Path<String>,
    ctx: web::Data<crate::app_context::AppContext>,
) -> Result<HttpResponse> {
    controller::unload_collection(collection_id, ctx).await
}

pub(crate) async fn get_loaded_collections(
    ctx: web::Data<crate::app_context::AppContext>,
) -> Result<HttpResponse> {
    controller::get_loaded_collections(ctx).await
}
