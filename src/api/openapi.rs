use utoipa::OpenApi;

/// API documentation for authentication endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::auth::controller::create_session
    ),
    components(
        schemas(
            crate::api::auth::dtos::CreateSessionDTO,
            crate::api::auth::dtos::Session,
            crate::api::auth::dtos::Claims
        )
    ),
    tags(
        (name = "auth", description = "Authentication endpoints")
    ),
    info(
        title = "Cosdata API",
        version = env!("CARGO_PKG_VERSION"),
        description = "Cosdata Vector Database API - Authentication",
        license(
            name = "Apache 2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    )
)]
pub struct AuthApiDoc;

/// API documentation for collection management endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::vectordb::collections::controller::create_collection,
        crate::api::vectordb::collections::controller::get_collections,
        crate::api::vectordb::collections::controller::get_collection_by_id,
        crate::api::vectordb::collections::controller::get_collection_indexing_status,
        crate::api::vectordb::collections::controller::delete_collection_by_id,
        crate::api::vectordb::collections::controller::load_collection,
        crate::api::vectordb::collections::controller::unload_collection,
        crate::api::vectordb::collections::controller::get_loaded_collections,
        crate::api::vectordb::collections::controller::list_collections
    ),
    components(
        schemas(
            crate::api::vectordb::collections::dtos::CreateCollectionDto,
            crate::api::vectordb::collections::dtos::CreateCollectionDtoResponse,
            crate::api::vectordb::collections::dtos::GetCollectionsDto,
            crate::api::vectordb::collections::dtos::GetCollectionsResponseDto,
            crate::api::vectordb::collections::dtos::ListCollectionsResponseDto,
            crate::api::vectordb::collections::dtos::CollectionSummaryDto,
            crate::api::vectordb::collections::dtos::MetadataField,
            crate::api::vectordb::collections::dtos::MetadataSchemaParam,
            crate::api::vectordb::collections::dtos::SupportedCondition,
            crate::api::vectordb::collections::dtos::ConditionOp,
            crate::models::collection::CollectionConfig,
            crate::models::collection::DenseVectorOptions,
            crate::models::collection::SparseVectorOptions,
            crate::models::collection::TFIDFOptions,
            CollectionIndexingStatusResponse
        )
    ),
    tags(
        (name = "collections", description = "Collection management endpoints")
    ),
    info(
        title = "Cosdata API",
        version = env!("CARGO_PKG_VERSION"),
        description = "Cosdata Vector Database API - Collection Management",
        license(
            name = "Apache 2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    )
)]
pub struct CollectionsApiDoc;

/// Combined API documentation
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::auth::controller::create_session,
        crate::api::vectordb::collections::controller::create_collection,
        crate::api::vectordb::collections::controller::get_collections,
        crate::api::vectordb::collections::controller::get_collection_by_id,
        crate::api::vectordb::collections::controller::get_collection_indexing_status,
        crate::api::vectordb::collections::controller::delete_collection_by_id,
        crate::api::vectordb::collections::controller::load_collection,
        crate::api::vectordb::collections::controller::unload_collection,
        crate::api::vectordb::collections::controller::get_loaded_collections,
        crate::api::vectordb::collections::controller::list_collections
    ),
    components(
        schemas(
            crate::api::auth::dtos::CreateSessionDTO,
            crate::api::auth::dtos::Session,
            crate::api::auth::dtos::Claims,
            crate::api::vectordb::collections::dtos::CreateCollectionDto,
            crate::api::vectordb::collections::dtos::CreateCollectionDtoResponse,
            crate::api::vectordb::collections::dtos::GetCollectionsDto,
            crate::api::vectordb::collections::dtos::GetCollectionsResponseDto,
            crate::api::vectordb::collections::dtos::ListCollectionsResponseDto,
            crate::api::vectordb::collections::dtos::CollectionSummaryDto,
            crate::api::vectordb::collections::dtos::MetadataField,
            crate::api::vectordb::collections::dtos::MetadataSchemaParam,
            crate::api::vectordb::collections::dtos::SupportedCondition,
            crate::api::vectordb::collections::dtos::ConditionOp,
            crate::models::collection::CollectionConfig,
            crate::models::collection::DenseVectorOptions,
            crate::models::collection::SparseVectorOptions,
            crate::models::collection::TFIDFOptions,
            CollectionIndexingStatusResponse
        )
    ),
    tags(
        (name = "auth", description = "Authentication endpoints"),
        (name = "collections", description = "Collection management endpoints")
    ),
    info(
        title = "Cosdata API",
        version = env!("CARGO_PKG_VERSION"),
        description = "Cosdata Vector Database API",
        license(
            name = "Apache 2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    )
)]
pub struct CombinedApiDoc;

/// Simplified schema for collection indexing status to avoid complex type issues
#[derive(utoipa::ToSchema, serde::Serialize)]
pub struct CollectionIndexingStatusResponse {
    pub collection_name: String,
    pub total_transactions: u32,
    pub completed_transactions: u32,
    pub in_progress_transactions: u32,
    pub not_started_transactions: u32,
    pub total_records_indexed_completed: u64,
    pub average_rate_per_second_completed: f32,
    pub last_synced: String,
}