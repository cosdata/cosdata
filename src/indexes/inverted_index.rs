use std::{
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicUsize},
        mpsc, Arc, RwLock,
    },
    thread,
};

use arcshift::ArcShift;
use lmdb::{Transaction, WriteFlags};

use crate::{
    macros::key,
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        common::WaCustomError,
        embedding_persist::{write_sparse_embedding, EmbeddingOffset},
        types::{MetaDb, SparseVector, VectorId},
        versioning::{Hash, Version, VersionControl},
    },
    storage::inverted_index_sparse_ann_basic::{
        InvertedIndexSparseAnnBasicTSHashmap, InvertedIndexSparseAnnNodeBasicTSHashmap,
    },
};

use super::inverted_index_types::{RawSparseVectorEmbedding, SparsePair};

#[derive(Default)]
pub struct SamplingData {
    pub above_1: AtomicUsize,
    pub above_9: AtomicUsize,
    pub above_8: AtomicUsize,
    pub above_7: AtomicUsize,
    pub above_6: AtomicUsize,
    pub above_5: AtomicUsize,
    pub above_4: AtomicUsize,
    pub above_3: AtomicUsize,
    pub above_2: AtomicUsize,
    pub values_collected: AtomicUsize,
}

pub struct InvertedIndex {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub metadata_schema: Option<String>, //object (optional)
    pub max_vectors: Option<i32>,
    pub root: Arc<InvertedIndexSparseAnnBasicTSHashmap>,
    pub lmdb: MetaDb,
    pub current_version: ArcShift<Hash>,
    pub current_open_transaction: AtomicPtr<InvertedIndexTransaction>,
    pub vcs: Arc<VersionControl>,
    pub values_upper_bound: RwLock<f32>,
    pub is_configured: AtomicBool,
    pub sampling_data: SamplingData,
    pub vectors: RwLock<Vec<(VectorId, Vec<SparsePair>)>>,
    pub vectors_collected: AtomicUsize,
    pub sample_threshold: usize,
    pub vec_raw_manager: Arc<BufferManagerFactory<Hash>>,
    pub search_threshold: f32,
}

unsafe impl Send for InvertedIndex {}
unsafe impl Sync for InvertedIndex {}

impl InvertedIndex {
    pub fn new(
        name: String,
        description: Option<String>,
        root_path: PathBuf,
        auto_create_index: bool,
        metadata_schema: Option<String>,
        max_vectors: Option<i32>,
        lmdb: MetaDb,
        current_version: Hash,
        vcs: Arc<VersionControl>,
        vec_raw_manager: Arc<BufferManagerFactory<Hash>>,
        quantization_bits: u8,
        sample_threshold: usize,
        search_threshold: f32,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let root = Arc::new(InvertedIndexSparseAnnBasicTSHashmap::new(
            root_path,
            quantization_bits,
            current_version,
            data_file_parts,
        )?);

        Ok(Self {
            name,
            auto_create_index,
            description,
            max_vectors,
            metadata_schema,
            root,
            lmdb,
            current_version: ArcShift::new(current_version),
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            vcs,
            values_upper_bound: RwLock::new(1.0),
            is_configured: AtomicBool::new(false),
            sampling_data: SamplingData::default(),
            vectors: RwLock::new(Vec::new()),
            vectors_collected: AtomicUsize::new(0),
            sample_threshold,
            vec_raw_manager,
            search_threshold,
        })
    }

    /// Finds the node at a given dimension
    ///
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<&InvertedIndexSparseAnnNodeBasicTSHashmap> {
        self.root.find_node(dim_index)
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

    /// Adds a sparse vector to the index.
    pub fn add_sparse_vector(&self, vector: SparseVector, version: Hash) -> Result<(), BufIoError> {
        self.root
            .add_sparse_vector(vector, version, *self.values_upper_bound.read().unwrap())
    }

    // Get method
    pub fn get_current_version(&self) -> Hash {
        let mut arc = self.current_version.clone();
        arc.get().clone()
    }

    // Set method
    pub fn set_current_version(&self, new_version: Hash) {
        let mut arc = self.current_version.clone();
        arc.update(new_version);
    }
}

pub struct InvertedIndexTransaction {
    pub id: Hash,
    pub version_number: u16,
    raw_embedding_serializer_thread_handle: thread::JoinHandle<Result<(), WaCustomError>>,
    pub raw_embedding_channel: mpsc::Sender<RawSparseVectorEmbedding>,
}

impl InvertedIndexTransaction {
    pub fn new(dense_index: Arc<InvertedIndex>) -> Result<Self, WaCustomError> {
        let branch_info = dense_index
            .vcs
            .get_branch_info("main")
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get main branch info: {}", err))
            })?
            .unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = dense_index
            .vcs
            .generate_hash("main", Version::from(version_number))
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get transaction hash: {}", err))
            })?;

        let (raw_embedding_channel, rx) = mpsc::channel();

        let raw_embedding_serializer_thread_handle = {
            let bufman = dense_index.vec_raw_manager.get(id)?;

            thread::spawn(move || {
                let mut offsets = Vec::new();
                for raw_emb in rx {
                    let offset = write_sparse_embedding(bufman.clone(), &raw_emb)?;
                    let embedding_key = key!(e:raw_emb.hash_vec);
                    offsets.push((embedding_key, offset));
                }

                let env = dense_index.lmdb.env.clone();
                let db = dense_index.lmdb.db.clone();

                let mut txn = env.begin_rw_txn().map_err(|e| {
                    WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e))
                })?;
                for (key, offset) in offsets {
                    let offset = EmbeddingOffset {
                        version: id,
                        offset,
                    };
                    let offset_serialized = offset.serialize();

                    txn.put(*db, &key, &offset_serialized, WriteFlags::empty())
                        .map_err(|e| {
                            WaCustomError::DatabaseError(format!("Failed to put data: {}", e))
                        })?;
                }

                txn.commit().map_err(|e| {
                    WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
                })?;
                bufman.flush()?;
                Ok(())
            })
        };

        Ok(Self {
            id,
            raw_embedding_channel,
            raw_embedding_serializer_thread_handle,
            version_number: version_number as u16,
        })
    }

    pub fn post_raw_embedding(&self, raw_emb: RawSparseVectorEmbedding) {
        self.raw_embedding_channel.send(raw_emb).unwrap();
    }

    pub fn pre_commit(self, inverted_index: &InvertedIndex) -> Result<(), WaCustomError> {
        drop(self.raw_embedding_channel);
        inverted_index.root.serialize()?;
        inverted_index.root.cache.dim_bufman.flush()?;
        inverted_index.root.cache.data_bufmans.flush_all()?;
        self.raw_embedding_serializer_thread_handle
            .join()
            .unwrap()?;
        Ok(())
    }
}
