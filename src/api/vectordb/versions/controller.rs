use actix_web::{web, HttpResponse, Result};
use crate::app_context::AppContext;
use super::{dtos::{VersionListResponse, VersionMetadata}, service};

pub(crate) async fn list_versions(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let versions = service::list_versions(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(versions))
}

pub(crate) async fn get_current_version(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let current_version = service::get_current_version(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(current_version))
}

pub(crate) async fn set_current_version(
    path: web::Path<(String, String)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let (collection_id, version_hash) = path.into_inner();
    service::set_current_version(ctx.into_inner(), &collection_id, &version_hash).await?;
    Ok(HttpResponse::Ok().finish())
}

