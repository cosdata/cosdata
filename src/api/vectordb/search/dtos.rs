use serde::{Deserialize, Serialize};
use crate::metadata::query_filtering::Filter;
use crate::models::types::VectorId;
use crate::indexes::inverted::types::SparsePair;


fn default_top_k() -> usize { 10 }
fn default_fusion_constant_k() -> f32 { 60.0 }


#[derive(Deserialize, Debug)]
pub(crate) struct DenseSearchRequestDto {
    pub query_vector: Vec<f32>,
    pub top_k: Option<usize>,
    pub filter: Option<Filter>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct BatchDenseSearchRequestDto {
    pub query_vectors: Vec<Vec<f32>>,
    pub top_k: Option<usize>,
    pub filter: Option<Filter>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct SparseSearchRequestDto {
    pub query_terms: Vec<SparsePair>,
    pub top_k: Option<usize>,
    pub early_terminate_threshold: Option<f32>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct BatchSparseSearchRequestDto {
    pub query_terms_list: Vec<Vec<SparsePair>>,
    pub top_k: Option<usize>,
    pub early_terminate_threshold: Option<f32>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct HybridSearchRequestDto {
    pub query_vector: Vec<f32>,
    pub query_terms: Vec<SparsePair>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_fusion_constant_k")]
    pub fusion_constant_k: f32,
}


#[derive(Serialize, Debug, Clone)]
pub(crate) struct SearchResultItemDto {
    pub id: VectorId,
    pub score: f32,
}

#[derive(Serialize, Debug)]
pub(crate) struct SearchResponseDto {
    pub results: Vec<SearchResultItemDto>,
}

pub(crate) type BatchSearchResponseDto = Vec<SearchResponseDto>;

#[derive(Deserialize, Debug)]
pub(crate) struct FindSimilarSparseIdfDocumentDto {
    pub query: String,
    pub top_k: Option<usize>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct BatchSearchSparseIdfDocumentsDto {
    pub queries: Vec<String>,
    pub top_k: Option<usize>,
}
