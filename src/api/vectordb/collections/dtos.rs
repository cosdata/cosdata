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
pub(crate) struct CreateCollectionDto {
    pub vector_db_name: String,
    pub dimensions: i32,
    pub max_val: Option<f32>,
    pub min_val: Option<f32>,
}

#[derive(Serialize)]
pub(crate) struct CreateCollectionDtoResponse {
    pub id: String,
    pub name: String,
    pub dimensions: usize,
    pub min_val: Option<f32>,
    pub max_val: Option<f32>,
    // pub created_at: String, // will be added when vector store has a creation timestamp
}
