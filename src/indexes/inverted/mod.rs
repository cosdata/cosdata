pub(crate) mod data;
pub(crate) mod transaction;
pub(crate) mod types;
use crate::macros::key;
use lmdb::Transaction;

use std::{
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicUsize},
        RwLock,
    },
};

use transaction::InvertedIndexTransaction;
use types::{RawSparseVectorEmbedding, SamplingData, SparsePair};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    inverted_index::InvertedIndexRoot,
    tree_map::TreeMap,
    types::{MetaDb, VectorId},
    versioning::{Hash, VersionControl},
};

pub struct InvertedIndex {
    pub name: String,
    pub description: Option<String>,
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
    pub vec_raw_manager: BufferManagerFactory<u8>,
    pub vec_raw_map: TreeMap<RawSparseVectorEmbedding>,
}

unsafe impl Send for InvertedIndex {}
unsafe impl Sync for InvertedIndex {}

impl InvertedIndex {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        description: Option<String>,
        root_path: PathBuf,
        metadata_schema: Option<String>,
        max_vectors: Option<i32>,
        lmdb: MetaDb,
        current_version: Hash,
        vcs: VersionControl,
        vec_raw_manager: BufferManagerFactory<u8>,
        quantization_bits: u8,
        sample_threshold: usize,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexRoot::new(root_path, quantization_bits, data_file_parts)?;

        Ok(Self {
            name,
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
            vec_raw_map: TreeMap::new(),
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

    pub fn contains_vector_id(&self, vector_id_u32: u32) -> bool {
        let env = self.lmdb.env.clone();
        let db = *self.lmdb.db;
        let txn = match env.begin_ro_txn() {
            Ok(txn) => txn,
            Err(e) => {
                log::error!(
                    "LMDB RO txn failed for sparse contains_vector_id check: {}",
                    e
                );
                return false;
            }
        };

        let vector_id_obj = VectorId(vector_id_u32 as u64);
        let embedding_key = key!(e: &vector_id_obj);

        let found = match txn.get(db, &embedding_key) {
            Ok(_) => true,
            Err(lmdb::Error::NotFound) => false,
            Err(e) => {
                log::error!(
                    "LMDB error during sparse contains_vector_id get for {}: {}",
                    vector_id_u32,
                    e
                );
                false
            }
        };

        txn.abort();
        found
    }
}
