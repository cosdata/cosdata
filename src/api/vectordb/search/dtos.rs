use crate::metadata::query_filtering::Filter;
use crate::models::types::VectorId;
use crate::{indexes::inverted::types::SparsePair, models::types::DocumentId};
use serde::{Deserialize, Serialize};

fn default_top_k() -> usize {
    10
}

fn default_fusion_constant_k() -> f32 {
    60.0
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct DenseSearchRequestDto {
    pub query_vector: Vec<f32>,
    pub top_k: Option<usize>,
    #[schema(value_type = Option<String>)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub return_raw_text: bool,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct BatchDenseSearchRequestQueryDto {
    pub vector: Vec<f32>,
    #[schema(value_type = Option<String>)]
    pub filter: Option<Filter>,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct BatchDenseSearchRequestDto {
    pub queries: Vec<BatchDenseSearchRequestQueryDto>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub return_raw_text: bool,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct SparseSearchRequestDto {
    #[schema(value_type = Vec<String>)]
    pub query_terms: Vec<SparsePair>,
    pub top_k: Option<usize>,
    pub early_terminate_threshold: Option<f32>,
    #[serde(default)]
    pub return_raw_text: bool,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct BatchSparseSearchRequestDto {
    #[schema(value_type = Vec<Vec<String>>)]
    pub query_terms_list: Vec<Vec<SparsePair>>,
    pub top_k: Option<usize>,
    pub early_terminate_threshold: Option<f32>,
    #[serde(default)]
    pub return_raw_text: bool,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct HybridSearchRequestDto {
    #[serde(flatten)]
    pub query: HybridSearchQuery,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_fusion_constant_k")]
    pub fusion_constant_k: f32,
    #[serde(default)]
    pub return_raw_text: bool,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
#[serde(untagged)]
pub(crate) enum HybridSearchQuery {
    DenseAndSparse {
        query_vector: Vec<f32>,
        #[schema(value_type = Vec<String>)]
        query_terms: Vec<SparsePair>,
        sparse_early_terminate_threshold: Option<f32>,
    },
    DenseAndTFIDF {
        query_vector: Vec<f32>,
        query_text: String,
    },
    SparseAndTFIDF {
        #[schema(value_type = Vec<String>)]
        query_terms: Vec<SparsePair>,
        query_text: String,
        sparse_early_terminate_threshold: Option<f32>,
    },
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct BatchHybridSearchRequestDto {
    pub queries: Vec<HybridSearchQuery>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_fusion_constant_k")]
    pub fusion_constant_k: f32,
    #[serde(default)]
    pub return_raw_text: bool,
}

#[derive(Serialize, Debug, Clone, utoipa::ToSchema)]
pub(crate) struct SearchResultItemDto {
    #[schema(value_type = String)]
    pub id: VectorId,
    #[schema(value_type = Option<String>)]
    pub document_id: Option<DocumentId>,
    pub score: f32,
    pub text: Option<String>,
}

#[derive(Serialize, Debug, utoipa::ToSchema)]
pub(crate) struct SearchResponseDto {
    pub results: Vec<SearchResultItemDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

#[derive(Serialize, Debug, utoipa::ToSchema)]
pub(crate) struct BatchSearchResponseDto {
    pub responses: Vec<SearchResponseDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct FindSimilarTFIDFDocumentDto {
    pub query: String,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub return_raw_text: bool,
}

#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub(crate) struct BatchSearchTFIDFDocumentsDto {
    pub queries: Vec<String>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub return_raw_text: bool,
}
