use actix_web::{web, HttpResponse, Result};

use crate::app_context::AppContext;

use super::dtos::{CreateTFIDFIndexDto, IndexDetailsDto, IndexResponseDto, IndexType};
use super::error::IndexesError;
use super::{
    dtos::{CreateDenseIndexDto, CreateSparseIndexDto},
    service,
};

/// Create a dense vector index for a collection
///
/// Creates a new dense vector index for the specified collection
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/indexes/dense",
    request_body = CreateDenseIndexDto,
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 201, description = "Dense index created successfully", body = IndexResponseDto),
        (status = 400, description = "Bad request", body = serde_json::Value),
        (status = 404, description = "Collection not found", body = serde_json::Value),
        (status = 409, description = "Index already exists", body = serde_json::Value),
        (status = 500, description = "Internal server error", body = serde_json::Value)
    ),
    tag = "indexes"
)]
pub(crate) async fn create_dense_index(
    web::Json(create_index_dto): web::Json<CreateDenseIndexDto>,
    ctx: web::Data<AppContext>,
    collection_id: web::Path<String>,
) -> Result<HttpResponse> {
    service::create_dense_index(
        collection_id.into_inner(),
        create_index_dto,
        ctx.into_inner(),
    )
    .await?;
    Ok(HttpResponse::Created().json(IndexResponseDto {
        message: "Dense index created successfully".to_string(),
    }))
}

/// Create a sparse vector index for a collection
///
/// Creates a new sparse vector index for the specified collection
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/indexes/sparse",
    request_body = CreateSparseIndexDto,
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 201, description = "Sparse index created successfully", body = IndexResponseDto),
        (status = 400, description = "Bad request", body = serde_json::Value),
        (status = 404, description = "Collection not found", body = serde_json::Value),
        (status = 409, description = "Index already exists", body = serde_json::Value),
        (status = 500, description = "Internal server error", body = serde_json::Value)
    ),
    tag = "indexes"
)]
pub(crate) async fn create_sparse_index(
    web::Json(create_index_dto): web::Json<CreateSparseIndexDto>,
    ctx: web::Data<AppContext>,
    collection_id: web::Path<String>,
) -> Result<HttpResponse> {
    service::create_sparse_index(
        collection_id.into_inner(),
        create_index_dto,
        ctx.into_inner(),
    )
    .await?;
    Ok(HttpResponse::Created().json(IndexResponseDto {
        message: "Sparse index created successfully".to_string(),
    }))
}

/// Create a TF-IDF index for a collection
///
/// Creates a new TF-IDF index for the specified collection
#[utoipa::path(
    post,
    path = "/vectordb/collections/{collection_id}/indexes/tf-idf",
    request_body = CreateTFIDFIndexDto,
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 201, description = "TF-IDF index created successfully", body = IndexResponseDto),
        (status = 400, description = "Bad request", body = serde_json::Value),
        (status = 404, description = "Collection not found", body = serde_json::Value),
        (status = 409, description = "Index already exists", body = serde_json::Value),
        (status = 500, description = "Internal server error", body = serde_json::Value)
    ),
    tag = "indexes"
)]
pub(crate) async fn create_tf_idf_index(
    web::Json(create_index_dto): web::Json<CreateTFIDFIndexDto>,
    ctx: web::Data<AppContext>,
    collection_id: web::Path<String>,
) -> Result<HttpResponse> {
    service::create_tf_idf_index(
        collection_id.into_inner(),
        create_index_dto,
        ctx.into_inner(),
    )
    .await?;
    Ok(HttpResponse::Created().json(IndexResponseDto {
        message: "TF-IDF index created successfully".to_string(),
    }))
}

/// Get indexes for a collection
///
/// Retrieves all indexes associated with the specified collection
#[utoipa::path(
    get,
    path = "/vectordb/collections/{collection_id}/indexes",
    params(
        ("collection_id" = String, Path, description = "Collection identifier")
    ),
    responses(
        (status = 200, description = "Indexes retrieved successfully", body = IndexDetailsDto),
        (status = 404, description = "Collection not found", body = serde_json::Value),
        (status = 500, description = "Internal server error", body = serde_json::Value)
    ),
    tag = "indexes"
)]
pub(crate) async fn get_index(
    collection_id: web::Path<String>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, IndexesError> {
    let index_details = service::get_index(collection_id.into_inner(), ctx.into_inner()).await?;
    Ok(HttpResponse::Ok().json(index_details))
}

/// Delete an index from a collection
///
/// Deletes the specified index from a collection
#[utoipa::path(
    delete,
    path = "/vectordb/collections/{collection_id}/indexes/{index_type}",
    params(
        ("collection_id" = String, Path, description = "Collection identifier"),
        ("index_type" = IndexType, Path, description = "Type of index to delete (dense, sparse or tf_idf)")
    ),
    responses(
        (status = 204, description = "Index successfully deleted"),
        (status = 404, description = "Collection or index not found", body = serde_json::Value),
        (status = 500, description = "Internal server error", body = serde_json::Value)
    ),
    tag = "indexes"
)]
pub(crate) async fn delete_index(
    path: web::Path<(String, IndexType)>,
    ctx: web::Data<AppContext>,
) -> Result<HttpResponse, IndexesError> {
    let (collection_id, index_type) = path.into_inner();
    service::delete_index(collection_id, index_type, ctx.into_inner()).await?;
    Ok(HttpResponse::NoContent().finish())
}
