use std::{
    fs::OpenOptions,
    ptr,
    sync::{atomic::AtomicPtr, mpsc, Arc, RwLock},
    thread,
};

use arcshift::ArcShift;
use lmdb::{Transaction, WriteFlags};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    macros::key,
    models::{
        buffered_io::BufferManagerFactory,
        cache_loader::{NodeRegistry, ProbCache},
        common::WaCustomError,
        embedding_persist::{write_sparse_embedding, EmbeddingOffset},
        types::{MetaDb, SparseVector},
        versioning::{Hash, Version, VersionControl},
    },
    storage::inverted_index_sparse_ann_basic::{
        calculate_path, InvertedIndexSparseAnnNodeBasicTSHashmap,
    },
};

use super::inverted_index_types::RawSparseVectorEmbedding;

pub struct InvertedIndex {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub metadata_schema: Option<String>, //object (optional)
    pub max_vectors: Option<i32>,
    pub root: Arc<InvertedIndexSparseAnnNodeBasicTSHashmap>,
    pub cache: Arc<NodeRegistry>,
    pub prob_cache: Arc<ProbCache>,
    pub lmdb: MetaDb,
    pub current_version: ArcShift<Hash>,
    pub current_open_transaction: AtomicPtr<InvertedIndexTransaction>,
    pub vcs: Arc<VersionControl>,
    pub vec_raw_manager: Arc<BufferManagerFactory<Hash>>,
    pub index_manager: Arc<BufferManagerFactory<Hash>>,
}

unsafe impl Send for InvertedIndex {}
unsafe impl Sync for InvertedIndex {}

impl InvertedIndex {
    pub fn new(
        name: String,
        description: Option<String>,
        auto_create_index: bool,
        metadata_schema: Option<String>,
        max_vectors: Option<i32>,
        lmdb: MetaDb,
        current_version: ArcShift<Hash>,
        vcs: Arc<VersionControl>,
        vec_raw_manager: Arc<BufferManagerFactory<Hash>>,
        index_manager: Arc<BufferManagerFactory<Hash>>,
        quantization: u8,
    ) -> Self {
        let root = Arc::new(InvertedIndexSparseAnnNodeBasicTSHashmap::new(
            0,
            false,
            quantization,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, index_manager.clone()));
        let prop_file = Arc::new(RwLock::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(format!("collections/{}/prop.data", name))
                .unwrap(),
        ));
        let prob_cache = Arc::new(ProbCache::new(
            index_manager.clone(),
            index_manager.clone(),
            prop_file,
        ));

        InvertedIndex {
            name,
            auto_create_index,
            description,
            max_vectors,
            metadata_schema,
            root,
            cache,
            prob_cache,
            lmdb,
            current_version,
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            vcs,
            vec_raw_manager,
            index_manager,
        }
    }

    /// Finds the node at a given dimension
    ///
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<&InvertedIndexSparseAnnNodeBasicTSHashmap> {
        let mut current_node = &*self.root;
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            let node_res = unsafe { &*child }.try_get_data(&self.prob_cache).unwrap();
            current_node = node_res
        }

        Some(current_node)
    }

    /// Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = self.root.find_or_create_node(&path, &self.prob_cache);
        // value will be quantized while being inserted into the Node.
        node.insert(value, vector_id);
    }

    /// Adds a sparse vector to the index.
    pub fn add_sparse_vector(&self, vector: SparseVector) -> Result<(), String> {
        let vector_id = vector.vector_id;
        vector.entries.par_iter().for_each(|(dim_index, value)| {
            if *value != 0.0 {
                self.insert(*dim_index, *value, vector_id);
            }
        });
        Ok(())
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

    pub fn pre_commit(self) -> Result<(), WaCustomError> {
        drop(self.raw_embedding_channel);
        self.raw_embedding_serializer_thread_handle
            .join()
            .unwrap()?;
        Ok(())
    }
}
