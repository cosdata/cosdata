pub(crate) mod data;
pub(crate) mod transaction;

use std::{
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicPtr, AtomicU32, Ordering},
        Arc, RwLock,
    },
};

use transaction::InvertedIndexIDFTransaction;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    inverted_index_idf::InvertedIndexIDFRoot,
    tree_map::TreeMap,
    types::{MetaDb, VectorId},
    versioning::{Hash, VersionControl},
};

use super::inverted::types::{RawSparseVectorEmbedding, SparsePair};

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
    pub vec_raw_manager: BufferManagerFactory<u8>,
    pub vec_raw_map: TreeMap<RawSparseVectorEmbedding>,
    pub document_id_counter: AtomicU32,
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
        vec_raw_manager: BufferManagerFactory<u8>,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexIDFRoot::new(root_path, data_file_parts)?;

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
            vec_raw_manager,
            vec_raw_map: TreeMap::new(),
            document_id_counter: AtomicU32::new(0),
        })
    }

    pub fn insert(
        &self,
        version: Hash,
        ext_id: VectorId,
        terms: Vec<SparsePair>,
    ) -> Result<(), BufIoError> {
        self.root
            .total_documents_count
            .fetch_add(1, Ordering::Relaxed);

        let document_id = self.document_id_counter.fetch_add(1, Ordering::Relaxed);

        for SparsePair(term_hash, tf) in &terms {
            self.root.insert(*term_hash, *tf, document_id, version)?;
        }

        let vec_emb = RawSparseVectorEmbedding {
            raw_vec: Arc::new(terms),
            hash_vec: ext_id,
        };

        self.vec_raw_map
            .insert(version, document_id as u64, vec_emb);

        Ok(())
    }

    pub fn set_current_version(&self, version: Hash) {
        *self.current_version.write().unwrap() = version;
    }
}
