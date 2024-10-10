use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) struct FindCollectionDto {
    pub id: String,
    pub vector_db_name: String,
    pub dimensions: usize,
    // pub max_val: Option<f32>,
    // pub min_val: Option<f32>,
    // pub created_at: String, //vector stores doesn't store their time of creation
}

#[derive(Deserialize)]
pub(crate) struct DenseVectorOptions {
    pub enabled: bool,
    pub auto_create_index: bool,
    pub dimension: i32,
}

#[derive(Deserialize)]
pub(crate) struct SparseVectorOptions {
    pub enabled: bool,
    pub auto_create_index: bool,
}

#[derive(Deserialize)]
pub(crate) struct CreateCollectionConfig {
    pub max_vectors: Option<i32>,
    pub replication_factor: Option<i32>,
}

#[derive(Deserialize)]
pub(crate) struct CreateCollectionDto {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub metadata_schema: Option<String>, //object (optional)
    pub config: CreateCollectionConfig,
}

#[derive(Serialize)]
pub(crate) struct CreateCollectionDtoResponse {
    pub id: String,
    pub name: String,
    pub dimensions: usize,
}

#[derive(Deserialize)]
pub(crate) struct GetCollectionsDto {}
