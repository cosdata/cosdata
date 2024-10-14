use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub(crate) struct DenseVectorOptions {
    pub enabled: bool,
    pub auto_create_index: bool,
    pub dimension: i32,
}

#[derive(Deserialize, Clone)]
pub(crate) struct SparseVectorOptions {
    pub enabled: bool,
    pub auto_create_index: bool,
}

#[derive(Deserialize, Clone)]
pub(crate) struct CollectionConfig {
    pub max_vectors: Option<i32>,
    pub replication_factor: Option<i32>,
}

#[derive(Clone)]
pub(crate) struct Collection {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub metadata_schema: Option<String>, //object (optional)
    pub config: CollectionConfig,
}

impl Collection {
    pub fn new(
        name: &str,
        description: &Option<String>,
        dense_vector_options: &DenseVectorOptions,
        sparse_vector_options: &SparseVectorOptions,
        metadata_schema: &Option<String>,
        config: &CollectionConfig,
    ) -> Self {
        Collection {
            name: name.into(),
            description: description.clone(),
            dense_vector: dense_vector_options.clone(),
            sparse_vector: sparse_vector_options.clone(),
            metadata_schema: metadata_schema.clone(),
            config: config.clone(),
        }
    }
}
