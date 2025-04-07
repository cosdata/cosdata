pub(crate) mod data;
pub(crate) mod transaction;
pub(crate) mod types;

use std::{
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicUsize},
        RwLock,
    },
};

use transaction::InvertedIndexTransaction;
use types::{SamplingData, SparsePair};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    inverted_index::InvertedIndexRoot,
    types::{MetaDb, VectorId},
    versioning::{Hash, VersionControl},
};

pub struct InvertedIndex {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub metadata_schema: Option<String>, //object (optional)
    pub max_vectors: Option<i32>,
    pub root: InvertedIndexRoot,
    pub lmdb: MetaDb,
    pub current_version: RwLock<Hash>,
    pub current_open_transaction: AtomicPtr<InvertedIndexTransaction>,
    pub vcs: VersionControl,
    pub values_upper_bound: RwLock<f32>,
    pub is_configured: AtomicBool,
    pub sampling_data: SamplingData,
    pub vectors: RwLock<Vec<(VectorId, Vec<SparsePair>)>>,
    pub vectors_collected: AtomicUsize,
    pub sample_threshold: usize,
    pub vec_raw_manager: BufferManagerFactory<Hash>,
}

unsafe impl Send for InvertedIndex {}
unsafe impl Sync for InvertedIndex {}

impl InvertedIndex {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        description: Option<String>,
        root_path: PathBuf,
        auto_create_index: bool,
        metadata_schema: Option<String>,
        max_vectors: Option<i32>,
        lmdb: MetaDb,
        current_version: Hash,
        vcs: VersionControl,
        vec_raw_manager: BufferManagerFactory<Hash>,
        quantization_bits: u8,
        sample_threshold: usize,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexRoot::new(root_path, quantization_bits, data_file_parts)?;

        Ok(Self {
            name,
            auto_create_index,
            description,
            max_vectors,
            metadata_schema,
            root,
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            vcs,
            values_upper_bound: RwLock::new(1.0),
            is_configured: AtomicBool::new(false),
            sampling_data: SamplingData::default(),
            vectors: RwLock::new(Vec::new()),
            vectors_collected: AtomicUsize::new(0),
            sample_threshold,
            vec_raw_manager,
        })
    }

    /// Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(
        &self,
        dim_index: u32,
        value: f32,
        vector_id: u32,
        version: Hash,
    ) -> Result<(), BufIoError> {
        self.root.insert(
            dim_index,
            value,
            vector_id,
            version,
            *self.values_upper_bound.read().unwrap(),
        )
    }

    pub fn set_current_version(&self, new_version: Hash) {
        *self.current_version.write().unwrap() = new_version;
    }
}
