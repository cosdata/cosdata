use utoipa::{openapi, OpenApi};

fn api_info() -> openapi::Info {
    openapi::InfoBuilder::new()
        .title("Cosdata API")
        .version(env!("CARGO_PKG_VERSION"))
        .description(Some("Cosdata Vector Database API\n\nAPI description for Cosdata vector search engine.\n\nThis document describes CRUD and search operations on collections of points (vectors with payload)."))
        .contact(Some(
            openapi::ContactBuilder::new()
                .email(Some("info@cosdata.io"))
                .url(Some("https://cosdata.io"))
                .build(),
        ))
        .license(Some(
            openapi::LicenseBuilder::new()
                .name("Apache 2.0")
                .url(Some("https://www.apache.org/licenses/LICENSE-2.0"))
                .build(),
        ))
        .build()
}

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
    modifiers(&AuthApiDoc)
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
        crate::api::vectordb::collections::controller::get_loaded_collections
    ),
    components(
        schemas(
            crate::api::vectordb::collections::dtos::CreateCollectionDto,
            crate::api::vectordb::collections::dtos::CreateCollectionDtoResponse,
            crate::api::vectordb::collections::dtos::GetCollectionsDto,
            crate::api::vectordb::collections::dtos::GetCollectionsResponseDto,
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
    modifiers(&CollectionsApiDoc)
)]
pub struct CollectionsApiDoc;

/// API documentation for index management endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::vectordb::indexes::controller::create_dense_index,
        crate::api::vectordb::indexes::controller::create_sparse_index,
        crate::api::vectordb::indexes::controller::create_tf_idf_index,
        crate::api::vectordb::indexes::controller::get_index,
        crate::api::vectordb::indexes::controller::delete_index
    ),
    components(
        schemas(
            crate::api::vectordb::indexes::dtos::CreateDenseIndexDto,
            crate::api::vectordb::indexes::dtos::CreateSparseIndexDto,
            crate::api::vectordb::indexes::dtos::CreateTFIDFIndexDto,
            crate::api::vectordb::indexes::dtos::IndexType,
            crate::api::vectordb::indexes::dtos::SparseIndexQuantization,
            crate::api::vectordb::indexes::dtos::DataType,
            crate::api::vectordb::indexes::dtos::ValuesRange,
            crate::api::vectordb::indexes::dtos::DenseIndexQuantizationDto,
            crate::api::vectordb::indexes::dtos::HNSWHyperParamsDto,
            crate::api::vectordb::indexes::dtos::DenseIndexParamsDto,
            crate::models::schema_traits::DistanceMetricSchema,
            crate::api::vectordb::indexes::dtos::IndexResponseDto,
            crate::api::vectordb::indexes::dtos::IndexDetailsDto,
            crate::api::vectordb::indexes::dtos::IndexInfo,
            crate::api::vectordb::indexes::dtos::DenseIndexInfo,
            crate::api::vectordb::indexes::dtos::SparseIndexInfo,
            crate::api::vectordb::indexes::dtos::TfIdfIndexInfo,
            crate::api::vectordb::indexes::dtos::QuantizationInfo,
            crate::api::vectordb::indexes::dtos::RangeInfo,
            crate::api::vectordb::indexes::dtos::HnswParamsInfo
        )
    ),
    tags(
        (name = "indexes", description = "Index management endpoints")
    ),
    modifiers(&IndexesApiDoc)
)]
pub struct IndexesApiDoc;

/// API documentation for search endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::vectordb::search::controller::dense_search,
        crate::api::vectordb::search::controller::batch_dense_search,
        crate::api::vectordb::search::controller::sparse_search,
        crate::api::vectordb::search::controller::batch_sparse_search,
        crate::api::vectordb::search::controller::hybrid_search,
        crate::api::vectordb::search::controller::batch_hybrid_search,
        crate::api::vectordb::search::controller::tf_idf_search,
        crate::api::vectordb::search::controller::batch_tf_idf_search
    ),
    components(
        schemas(
            crate::api::vectordb::search::dtos::DenseSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchDenseSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchDenseSearchRequestQueryDto,
            crate::api::vectordb::search::dtos::SparseSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchSparseSearchRequestDto,
            crate::api::vectordb::search::dtos::HybridSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchHybridSearchRequestDto,
            crate::api::vectordb::search::dtos::HybridSearchQuery,
            crate::api::vectordb::search::dtos::FindSimilarTFIDFDocumentDto,
            crate::api::vectordb::search::dtos::BatchSearchTFIDFDocumentsDto,
            crate::api::vectordb::search::dtos::SearchResultItemDto,
            crate::api::vectordb::search::dtos::SearchResponseDto,
            crate::api::vectordb::search::dtos::BatchSearchResponseDto
        )
    ),
    tags(
        (name = "search", description = "Vector search endpoints")
    ),
    modifiers(&SearchApiDoc)
)]
pub struct SearchApiDoc;

/// API documentation for version management endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::vectordb::versions::controller::list_versions,
        crate::api::vectordb::versions::controller::get_current_version,
        crate::api::vectordb::versions::controller::set_current_version
    ),
    components(
        schemas(
            crate::api::vectordb::versions::dtos::VersionMetadata,
            crate::api::vectordb::versions::dtos::VersionListResponse,
            crate::api::vectordb::versions::dtos::CurrentVersionResponse
        )
    ),
    tags(
        (name = "versions", description = "Version management endpoints")
    ),
    modifiers(&VersionsApiDoc)
)]
pub struct VersionsApiDoc;

/// API documentation for transaction management endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::vectordb::transactions::controller::create_transaction,
        crate::api::vectordb::transactions::controller::commit_transaction,
        crate::api::vectordb::transactions::controller::get_transaction_status,
        crate::api::vectordb::transactions::controller::create_vector_in_transaction,
        crate::api::vectordb::transactions::controller::delete_vector_by_id,
        crate::api::vectordb::transactions::controller::abort_transaction,
        crate::api::vectordb::transactions::controller::upsert
    ),
    components(
        schemas(
            crate::api::vectordb::transactions::dtos::CreateTransactionResponseDto,
            crate::api::vectordb::transactions::dtos::UpsertDto,
            crate::models::collection_transaction::TransactionStatus,
            crate::models::collection_transaction::ProcessingStats
        )
    ),
    tags(
        (name = "transactions", description = "Transaction management endpoints")
    ),
    modifiers(&TransactionsApiDoc)
)]
pub struct TransactionsApiDoc;

/// API documentation for vector management endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::vectordb::vectors::controller::query_vectors,
        crate::api::vectordb::vectors::controller::get_vector_by_id,
        crate::api::vectordb::vectors::controller::check_vector_existence,
        crate::api::vectordb::vectors::controller::fetch_vector_neighbors
    ),
    components(
        schemas(
            crate::api::vectordb::vectors::dtos::VectorsQueryDto,
            crate::api::vectordb::vectors::dtos::CreateVectorDto,
            crate::api::vectordb::vectors::dtos::SimilarVector
        )
    ),
    tags(
        (name = "vectors", description = "Vector management endpoints")
    ),
    modifiers(&VectorsApiDoc)
)]
pub struct VectorsApiDoc;

/// API documentation for streaming endpoints
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::vectordb::streaming::controller::upsert,
        crate::api::vectordb::streaming::controller::delete_vector_by_id
    ),
    components(
        schemas(
            crate::api::vectordb::transactions::dtos::UpsertDto
        )
    ),
    tags(
        (name = "streaming", description = "Streaming endpoints")
    ),
    modifiers(&StreamingApiDoc)
)]
pub struct StreamingApiDoc;

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
        crate::api::vectordb::indexes::controller::create_dense_index,
        crate::api::vectordb::indexes::controller::create_sparse_index,
        crate::api::vectordb::indexes::controller::create_tf_idf_index,
        crate::api::vectordb::indexes::controller::get_index,
        crate::api::vectordb::indexes::controller::delete_index,
        crate::api::vectordb::search::controller::dense_search,
        crate::api::vectordb::search::controller::batch_dense_search,
        crate::api::vectordb::search::controller::sparse_search,
        crate::api::vectordb::search::controller::batch_sparse_search,
        crate::api::vectordb::search::controller::hybrid_search,
        crate::api::vectordb::search::controller::batch_hybrid_search,
        crate::api::vectordb::search::controller::tf_idf_search,
        crate::api::vectordb::search::controller::batch_tf_idf_search,
        crate::api::vectordb::vectors::controller::query_vectors,
        crate::api::vectordb::vectors::controller::get_vector_by_id,
        crate::api::vectordb::vectors::controller::check_vector_existence,
        crate::api::vectordb::vectors::controller::fetch_vector_neighbors,
        crate::api::vectordb::versions::controller::list_versions,
        crate::api::vectordb::versions::controller::get_current_version,
        crate::api::vectordb::versions::controller::set_current_version,
        crate::api::vectordb::transactions::controller::create_transaction,
        crate::api::vectordb::transactions::controller::commit_transaction,
        crate::api::vectordb::transactions::controller::get_transaction_status,
        crate::api::vectordb::transactions::controller::create_vector_in_transaction,
        crate::api::vectordb::transactions::controller::delete_vector_by_id,
        crate::api::vectordb::transactions::controller::abort_transaction,
        crate::api::vectordb::transactions::controller::upsert,
        crate::api::vectordb::streaming::controller::upsert,
        crate::api::vectordb::streaming::controller::delete_vector_by_id
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
            crate::api::vectordb::collections::dtos::MetadataField,
            crate::api::vectordb::collections::dtos::MetadataSchemaParam,
            crate::api::vectordb::collections::dtos::SupportedCondition,
            crate::api::vectordb::collections::dtos::ConditionOp,
            crate::models::collection::CollectionConfig,
            crate::models::collection::DenseVectorOptions,
            crate::models::collection::SparseVectorOptions,
            crate::models::collection::TFIDFOptions,
            CollectionIndexingStatusResponse,
            crate::api::vectordb::indexes::dtos::CreateDenseIndexDto,
            crate::api::vectordb::indexes::dtos::CreateSparseIndexDto,
            crate::api::vectordb::indexes::dtos::CreateTFIDFIndexDto,
            crate::api::vectordb::indexes::dtos::IndexType,
            crate::api::vectordb::indexes::dtos::SparseIndexQuantization,
            crate::api::vectordb::indexes::dtos::DataType,
            crate::api::vectordb::indexes::dtos::ValuesRange,
            crate::api::vectordb::indexes::dtos::DenseIndexQuantizationDto,
            crate::api::vectordb::indexes::dtos::HNSWHyperParamsDto,
            crate::api::vectordb::indexes::dtos::DenseIndexParamsDto,
            crate::models::schema_traits::DistanceMetricSchema,
            crate::api::vectordb::indexes::dtos::IndexResponseDto,
            crate::api::vectordb::indexes::dtos::IndexDetailsDto,
            crate::api::vectordb::indexes::dtos::IndexInfo,
            crate::api::vectordb::indexes::dtos::DenseIndexInfo,
            crate::api::vectordb::indexes::dtos::SparseIndexInfo,
            crate::api::vectordb::indexes::dtos::TfIdfIndexInfo,
            crate::api::vectordb::indexes::dtos::QuantizationInfo,
            crate::api::vectordb::indexes::dtos::RangeInfo,
            crate::api::vectordb::indexes::dtos::HnswParamsInfo,
            crate::api::vectordb::search::dtos::DenseSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchDenseSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchDenseSearchRequestQueryDto,
            crate::api::vectordb::search::dtos::SparseSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchSparseSearchRequestDto,
            crate::api::vectordb::search::dtos::HybridSearchRequestDto,
            crate::api::vectordb::search::dtos::BatchHybridSearchRequestDto,
            crate::api::vectordb::search::dtos::HybridSearchQuery,
            crate::api::vectordb::search::dtos::FindSimilarTFIDFDocumentDto,
            crate::api::vectordb::search::dtos::BatchSearchTFIDFDocumentsDto,
            crate::api::vectordb::search::dtos::SearchResultItemDto,
            crate::api::vectordb::search::dtos::SearchResponseDto,
            crate::api::vectordb::search::dtos::BatchSearchResponseDto,
            crate::api::vectordb::vectors::dtos::VectorsQueryDto,
            crate::api::vectordb::vectors::dtos::CreateVectorDto,
            crate::api::vectordb::vectors::dtos::SimilarVector,
            crate::api::vectordb::versions::dtos::VersionMetadata,
            crate::api::vectordb::versions::dtos::VersionListResponse,
            crate::api::vectordb::versions::dtos::CurrentVersionResponse,
            crate::api::vectordb::transactions::dtos::CreateTransactionResponseDto,
            crate::api::vectordb::transactions::dtos::UpsertDto,
            crate::models::collection_transaction::TransactionStatus,
            crate::models::collection_transaction::ProcessingStats
        )
    ),
    tags(
        (name = "auth", description = "Authentication endpoints"),
        (name = "collections", description = "Collection management endpoints"),
        (name = "indexes", description = "Index management endpoints"),
        (name = "search", description = "Vector search endpoints"),
        (name = "vectors", description = "Vector management endpoints"),
        (name = "versions", description = "Version management endpoints"),
        (name = "transactions", description = "Transaction management endpoints"),
        (name = "streaming", description = "Streaming endpoints")
    ),
    modifiers(&CombinedApiDoc)
)]
pub struct CombinedApiDoc;

impl utoipa::Modify for AuthApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for CollectionsApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for IndexesApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for SearchApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for VectorsApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for TransactionsApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for VersionsApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for StreamingApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

impl utoipa::Modify for CombinedApiDoc {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = api_info();
    }
}

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
