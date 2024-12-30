use std::sync::Arc;

use arcshift::ArcShift;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    models::{
        buffered_io::BufferManagerFactory,
        cache_loader::NodeRegistry,
        types::{MetaDb, SparseVector},
        versioning::{Hash, VersionControl},
    },
    storage::inverted_index_sparse_ann_basic::{
        calculate_path, InvertedIndexSparseAnnNodeBasicTSHashmap,
    },
};

#[derive(Clone)]
pub struct InvertedIndex {
    pub name: String,
    pub description: Option<String>,
    pub auto_create_index: bool,
    pub metadata_schema: Option<String>, //object (optional)
    pub max_vectors: Option<i32>,
    pub root: ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>,
    pub cache: Arc<NodeRegistry>,
    pub lmdb: MetaDb,
    pub current_version: ArcShift<Hash>,
    pub current_open_transaction: ArcShift<Option<Hash>>,
    pub vcs: Arc<VersionControl>,
    pub vec_raw_manager: Arc<BufferManagerFactory>,
    pub index_manager: Arc<BufferManagerFactory>,
}

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
        vec_raw_manager: Arc<BufferManagerFactory>,
        index_manager: Arc<BufferManagerFactory>,
    ) -> Self {
        let root = ArcShift::new(InvertedIndexSparseAnnNodeBasicTSHashmap::new(0, false));
        let cache = Arc::new(NodeRegistry::new(1000, index_manager.clone()));

        InvertedIndex {
            name,
            auto_create_index,
            description,
            max_vectors,
            metadata_schema,
            root,
            cache,
            lmdb,
            current_version,
            current_open_transaction: ArcShift::new(None),
            vcs,
            vec_raw_manager,
            index_manager,
        }
    }

    /// Finds the node at a given dimension
    ///
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(
        &self,
        dim_index: u32,
    ) -> Option<ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            let node_res = child.get_data(self.cache.clone());
            current_node = ArcShift::new((*node_res).clone());
        }

        Some(current_node)
    }

    /// Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    /// Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNodeBasicTSHashmap::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        // value will be quantized while being inserted into the Node.
        InvertedIndexSparseAnnNodeBasicTSHashmap::insert(node, value, vector_id)
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
