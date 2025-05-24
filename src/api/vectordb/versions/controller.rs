use super::dtos::{CurrentVersionResponse, VersionListResponse};
use super::error::VersionError;
use super::service;
use crate::app_context::AppContext;
use actix_web::{web, HttpResponse, Result};

/// List all versions of a collection
#[utoipa::path(
    get,
    path = "/vectordb/collections/{collection_id}/versions",
    tag = "versions",
    params(
        ("collection_id" = String, Path, description = "ID of the collection")
    ),
    responses(
        (status = 200, description = "List of collection versions", body = VersionListResponse),
        (status = 404, description = "Collection not found"),
        (status = 500, description = "Database error")
    )
)]
pub(crate) async fn list_versions(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VersionError> {
    let versions = service::list_versions(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(versions))
}

/// Get the current version of a collection
#[utoipa::path(
    get,
    path = "/vectordb/collections/{collection_id}/versions/current",
    tag = "versions",
    params(
        ("collection_id" = String, Path, description = "ID of the collection")
    ),
    responses(
        (status = 200, description = "Current collection version", body = CurrentVersionResponse),
        (status = 404, description = "Collection not found"),
        (status = 500, description = "Database error")
    )
)]
pub(crate) async fn get_current_version(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VersionError> {
    let current_version = service::get_current_version(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(current_version))
}

/// Set the current version of a collection
#[utoipa::path(
    put,
    path = "/vectordb/collections/{collection_id}/versions/current/{version_hash}",
    tag = "versions",
    params(
        ("collection_id" = String, Path, description = "ID of the collection"),
        ("version_hash" = String, Path, description = "Hash of the version to set as current")
    ),
    responses(
        (status = 200, description = "Version set as current successfully"),
        (status = 400, description = "Invalid version hash"),
        (status = 404, description = "Collection not found"),
        (status = 500, description = "Database error")
    )
)]
#[allow(unused)]
pub(crate) async fn set_current_version(
    path: web::Path<(String, String)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, VersionError> {
    let (collection_id, version_hash) = path.into_inner();
    service::set_current_version(ctx.into_inner(), &collection_id, &version_hash).await?;
    Ok(HttpResponse::Ok().finish())
}
