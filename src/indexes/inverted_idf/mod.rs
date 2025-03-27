pub(crate) mod data;
pub(crate) mod transaction;

use std::{
    path::PathBuf,
    ptr,
    sync::{atomic::AtomicPtr, RwLock},
};

use transaction::InvertedIndexIDFTransaction;

use crate::models::{
    buffered_io::BufIoError,
    inverted_index_idf::InvertedIndexIDFRoot,
    types::MetaDb,
    versioning::{Hash, VersionControl},
};

pub struct InvertedIndexIDF {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub max_vectors: Option<i32>,
    pub root: InvertedIndexIDFRoot,
    pub lmdb: MetaDb,
    pub current_version: RwLock<Hash>,
    pub current_open_transaction: AtomicPtr<InvertedIndexIDFTransaction>,
    pub vcs: VersionControl,
    pub early_terminate_threshold: f32,
}

unsafe impl Send for InvertedIndexIDF {}
unsafe impl Sync for InvertedIndexIDF {}

impl InvertedIndexIDF {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        description: Option<String>,
        root_path: PathBuf,
        auto_create_index: bool,
        max_vectors: Option<i32>,
        lmdb: MetaDb,
        current_version: Hash,
        vcs: VersionControl,
        quantization_bits: u8,
        data_file_parts: u8,
        early_terminate_threshold: f32,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexIDFRoot::new(root_path, quantization_bits, data_file_parts)?;

        Ok(Self {
            name,
            auto_create_index,
            description,
            max_vectors,
            root,
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            vcs,
            early_terminate_threshold,
        })
    }

    pub fn insert(
        &self,
        hash_dim: u32,
        value: f32,
        document_id: u32,
        version: Hash,
    ) -> Result<(), BufIoError> {
        self.root.insert(hash_dim, value, document_id, version)
    }

    pub fn set_current_version(&self, version: Hash) {
        *self.current_version.write().unwrap() = version;
    }
}
