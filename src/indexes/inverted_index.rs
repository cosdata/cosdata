use std::{
    fs::File,
    sync::{Arc, Mutex},
};

use arcshift::ArcShift;

use crate::{
    models::{
        types::{DistanceMetric, MetaDb, QuantizationMetric},
        versioning::{Hash, VersionControl},
    },
    quantization::StorageType,
};

use super::inverted_index_item::InvertedIndexItem;

#[allow(dead_code)]
pub(crate) struct InvertedIndex {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub metadata_schema: Option<String>, //object (optional)
    pub max_vectors: Option<i32>,
    pub replication_factor: Option<i32>,
    pub root: Arc<Mutex<InvertedIndexItem>>,
    pub prop_file: Arc<File>,
    pub lmdb: MetaDb,
    pub current_version: ArcShift<Hash>,
    pub current_open_transaction: ArcShift<Option<Hash>>,
    pub quantization_metric: Arc<QuantizationMetric>,
    pub distance_metric: Arc<DistanceMetric>,
    pub storage_type: StorageType,
    pub vcs: Arc<VersionControl>,
}

#[allow(dead_code)]
impl InvertedIndex {
    pub fn new(
        name: String,
        description: Option<String>,
        auto_create_index: bool,
        metadata_schema: Option<String>,
        max_vectors: Option<i32>,
        replication_factor: Option<i32>,
        prop_file: Arc<File>,
        lmdb: MetaDb,
        current_version: ArcShift<Hash>,
        quantization_metric: Arc<QuantizationMetric>,
        distance_metric: Arc<DistanceMetric>,
        storage_type: StorageType,
        vcs: Arc<VersionControl>,
    ) -> Self {
        InvertedIndex {
            name,
            auto_create_index,
            description,
            max_vectors,
            metadata_schema,
            replication_factor,
            root: Arc::new(Mutex::new(InvertedIndexItem::new(0, false))),
            prop_file,
            lmdb,
            current_version,
            current_open_transaction: ArcShift::new(None),
            quantization_metric,
            distance_metric,
            storage_type,
            vcs,
        }
    }

    pub fn add_dim_index(&self, dim_index: u32, value: f32, vector_id: u32) -> Result<(), String> {
        self.root
            .lock()
            .unwrap()
            .insert_dim_index(dim_index, value, vector_id)
    }

    pub fn print_tree(&self) {
        self.root.lock().unwrap().print_tree(0);
    }
}
