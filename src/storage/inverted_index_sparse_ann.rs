use core::array::from_fn;
use rayon::prelude::*;
use std::path::Path;

use std::sync::Arc;

use crate::models::{
    buffered_io::BufferManagerFactory,
    cache_loader::NodeRegistry,
    lazy_load::{LazyItem, LazyItemArray, LazyItemVec},
    types::SparseVector,
    versioning::Hash,
};

// TODO: Add more powers for larger jumps
// TODO: Or switch to dynamic calculation of power of max power of 4
const POWERS_OF_4: [u32; 8] = [1, 4, 16, 64, 256, 1024, 4096, 16384];

/// Returns the largest power of 4 that is less than or equal to `n`.
/// Iteratively multiplies by 4 until the result exceeds `n`.
pub fn largest_power_of_4_below(n: u32) -> (usize, u32) {
    assert_ne!(n, 0, "Cannot find largest power of 4 below 0");
    POWERS_OF_4
        .into_iter()
        .enumerate()
        .rev()
        .find(|&(_, pow4)| pow4 <= n)
        .unwrap()
}

/// Calculates the path from `current_dim_index` to `target_dim_index`.
/// Decomposes the difference into powers of 4 and returns the indices.
fn calculate_path(target_dim_index: u32, current_dim_index: u32) -> Vec<usize> {
    let mut path = Vec::new();
    let mut remaining = target_dim_index - current_dim_index;

    while remaining > 0 {
        let (child_index, pow_4) = largest_power_of_4_below(remaining);
        path.push(child_index);
        remaining -= pow_4;
    }

    path
}

/// [InvertedIndexSparseAnnNode] (earlier InvertedIndexItem) is a node in InvertedIndexSparseAnn structure
/// data in InvertedIndexSparseAnnNode holds list of Vec_Ids corresponding to the quantized u8 value (which is the index of array)
#[derive(Clone)]
pub struct InvertedIndexSparseAnnNode {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: Arc<[LazyItemVec<u32>; 64]>,
    pub lazy_children: LazyItemArray<InvertedIndexSparseAnnNode, 16>,
}

impl InvertedIndexSparseAnnNode {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        let data: Arc<[LazyItemVec<u32>; 64]> = Arc::new(from_fn(|_| LazyItemVec::new()));
        InvertedIndexSparseAnnNode {
            dim_index,
            implicit,
            data,
            lazy_children: LazyItemArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: Arc<InvertedIndexSparseAnnNode>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> Arc<InvertedIndexSparseAnnNode> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(
                0.into(),
                0,
                InvertedIndexSparseAnnNode::new(new_dim_index, true),
            );
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
                {
                    current_node = child.get_data(cache.clone());
                    break;
                }
            }
        }

        current_node
    }

    pub fn quantize(value: f32) -> u8 {
        ((value * 63.0).clamp(0.0, 63.0) as u8).min(63)
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Finds the quantized value and pushes the vec_Id in array at index = quantized_value
    pub fn insert(node: Arc<InvertedIndexSparseAnnNode>, value: f32, vector_id: u32) {
        let quantized_value = Self::quantize(value);
        let mut data: Arc<[LazyItemVec<u32>; 64]> = node.data.clone();

        // Insert into the specific LazyItemVec at the index quantized_value
        if let Some(lazy_item_vec) = Arc::make_mut(&mut data).get_mut(quantized_value as usize) {
            lazy_item_vec.push(LazyItem::new(0.into(), 0, vector_id));
        }
    }

    /// Retrieves a value from the index at the specified dimension index.
    /// Calculates the path and delegates to `get_value`.
    pub fn get(&self, dim_index: u32, vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        let path = calculate_path(dim_index, self.dim_index);
        self.get_value(&path, vector_id, cache)
    }

    /// Retrieves a value from the index following the specified path.
    /// Recursively traverses child nodes or searches the data vector.
    fn get_value(&self, path: &[usize], vector_id: u32, cache: Arc<NodeRegistry>) -> Option<u8> {
        match path.get(0) {
            Some(child_index) => self
                .lazy_children
                .get(*child_index)
                .map(|data| {
                    data.get_data(cache.clone())
                        .get_value(&path[1..], vector_id, cache)
                })
                .flatten(),
            None => {
                for (index, lazy_item_vec) in self.data.iter().enumerate() {
                    if lazy_item_vec
                        .iter()
                        .any(|item| *item.get_data(cache.clone()) == vector_id)
                    {
                        return Some(index as u8);
                    }
                }
                None
            }
        }
    }
}

/// [InvertedIndexSparseAnn] is a improved version which only holds quantized u8 values instead of f32 inside [InvertedIndexSparseAnnNode]
#[derive(Clone)]
pub struct InvertedIndexSparseAnn {
    pub root: Arc<InvertedIndexSparseAnnNode>,
    pub cache: Arc<NodeRegistry>,
}

impl InvertedIndexSparseAnn {
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            1.0,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndexSparseAnn {
            root: Arc::new(InvertedIndexSparseAnnNode::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<Arc<InvertedIndexSparseAnnNode>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            current_node = child.get_data(self.cache.clone());
        }

        Some(current_node)
    }

    //Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root.get(dim_index, vector_id, self.cache.clone())
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNode::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        //value will be quantized while being inserted into the Node.
        InvertedIndexSparseAnnNode::insert(node, value, vector_id)
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
}
