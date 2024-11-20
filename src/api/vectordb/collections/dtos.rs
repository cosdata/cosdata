use serde::{Deserialize, Serialize};

use crate::models::collection::{CollectionConfig, DenseVectorOptions, SparseVectorOptions};

#[derive(Deserialize)]
pub(crate) struct CreateCollectionDto {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub metadata_schema: Option<String>, //object (optional)
    pub config: CollectionConfig,
}

#[derive(Serialize)]
pub(crate) struct CreateCollectionDtoResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct GetCollectionsDto {
    
    pub name: Option<String>,
    pub description: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct GetCollectionsResponseDto {
    pub name:String ,
    pub description: Option<String>,
}
