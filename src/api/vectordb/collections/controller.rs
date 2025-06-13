use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;

use super::{
    dtos::{
        CreateCollectionDto, CreateCollectionDtoResponse, GetCollectionsDto,
        GetCollectionsResponseDto, CollectionWithVectorCountsDto,
    },
    service,
};
use crate::api::openapi::CollectionIndexingStatusResponse;
// use crate::api::vectordb::collections::error::CollectionsError;

/// Create a new collection
///
/// Creates a new vector collection with the specified configuration.
#[utoipa::path(
    post,
    path = "/vectordb/collections",
    request_body = CreateCollectionDto,
    responses(
        (status = 201, description = "Collection created successfully", body = CreateCollectionDtoResponse),
        (status = 400, description = "Invalid request"),
        (status = 409, description = "Collection already exists"),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn create_collection(
    web::Json(create_collection_dto): web::Json<CreateCollectionDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let create_collection_response_dto =
        service::create_collection(ctx.into_inner(), create_collection_dto).await?;

    Ok(HttpResponse::Created().json(create_collection_response_dto))
}

/// Get all collections
///
/// Returns a list of all collections.
#[utoipa::path(
    get,
    path = "/vectordb/collections",
    params(
        GetCollectionsDto
    ),
    responses(
        (status = 200, description = "List of collections", body = Vec<GetCollectionsResponseDto>),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn get_collections(
    web::Query(get_collections_dto): web::Query<GetCollectionsDto>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collections = service::get_collections(ctx.into_inner(), get_collections_dto).await?;
    Ok(HttpResponse::Ok().json(collections))
}

/// Get collection by ID
///
/// Returns a specific collection by its ID.
#[utoipa::path(
    get,
    path = "/vectordb/collections/{collection_id}",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 200, description = "Collection information with vector counts", body = CollectionWithVectorCountsDto),
        (status = 400, description = "Collection not found"),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn get_collection_by_id(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection_with_counts = service::get_collection_by_id(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(collection_with_counts))
}

/// Get collection indexing status
///
/// Returns the indexing status of a collection.
#[utoipa::path(
    get,
    path = "/vectordb/collections/{collection_id}/indexing_status",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 200, description = "Collection indexing status", body = CollectionIndexingStatusResponse),
        (status = 400, description = "Collection not found"),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn get_collection_indexing_status(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let status = service::get_collection_indexing_status(ctx.into_inner(), &collection_id).await?;

    // Convert to the simplified response format for the API docs
    let response = CollectionIndexingStatusResponse {
        collection_name: status.collection_name,
        total_transactions: status.status_summary.total_transactions,
        completed_transactions: status.status_summary.completed_transactions,
        in_progress_transactions: status.status_summary.in_progress_transactions,
        not_started_transactions: status.status_summary.not_started_transactions,
        total_records_indexed_completed: status.status_summary.total_records_indexed_completed,
        average_rate_per_second_completed: status.status_summary.average_rate_per_second_completed,
        last_synced: status.last_synced.to_rfc3339(),
    };

    Ok(HttpResponse::Ok().json(response))
}

/// Delete collection by ID
///
/// Deletes a collection by its ID.
#[utoipa::path(
    delete,
    path = "/vectordb/collections/{collection_id}",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 204, description = "Collection deleted successfully"),
        (status = 400, description = "Collection not found"),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn delete_collection_by_id(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    service::delete_collection_by_id(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::NoContent().finish())
}

/// Load collection into memory
///
/// Loads a collection into memory for faster access.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/load",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 200, description = "Collection loaded successfully"),
        (status = 400, description = "Collection not found"),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn load_collection(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    let collection = service::load_collection(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(&collection.meta))
}

/// Unload collection from memory
///
/// Unloads a collection from memory.
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/unload",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 200, description = "Collection unloaded successfully"),
        (status = 400, description = "Collection not found"),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn unload_collection(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse> {
    service::unload_collection(ctx.into_inner(), &collection_id).await?;
    Ok(HttpResponse::Ok().json(format!(
        "Collection '{}' successfully unloaded",
        collection_id
    )))
}

/// Get loaded collections
///
/// Returns a list of all collections currently loaded in memory.
#[utoipa::path(
    get,
    path = "/vectordb/collections/loaded",
    responses(
        (status = 200, description = "List of loaded collections", body = Vec<String>),
        (status = 500, description = "Server error")
    ),
    tag = "collections"
)]
pub(crate) async fn get_loaded_collections(ctx: web::Data<AppContext>) -> Result<HttpResponse> {
    let collections = service::get_loaded_collections(ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(collections))
}
