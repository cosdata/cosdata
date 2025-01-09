use arcshift::ArcShift;
use core::array::from_fn;
use dashmap::DashMap;
use rayon::prelude::*;
use std::{path::Path, sync::RwLock};

use std::sync::Arc;

use crate::models::cuckoo_filter_tree::CuckooFilterTreeNode;
use crate::models::versioning::Hash;
use crate::models::{
    buffered_io::BufferManagerFactory,
    cache_loader::NodeRegistry,
    common::TSHashTable,
    lazy_load::{LazyItem, LazyItemArray},
    types::SparseVector,
};

use super::page::Pagepool;

// Size of a page in the hash table
const PAGE_SIZE: usize = 32;

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
pub fn calculate_path(target_dim_index: u32, current_dim_index: u32) -> Vec<usize> {
    let mut path = Vec::new();
    let mut remaining = target_dim_index - current_dim_index;

    while remaining > 0 {
        let (child_index, pow_4) = largest_power_of_4_below(remaining);
        path.push(child_index);
        remaining -= pow_4;
    }

    path
}

/// [InvertedIndexSparseAnnNodeBasic] is a node in InvertedIndexSparseAnnBasic structure
/// data in InvertedIndexSparseAnnNode holds list of Vec_Ids corresponding to the quantized u8 value (which is the index of array)
#[derive(Clone)]
pub struct InvertedIndexSparseAnnNodeBasic {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: [Arc<RwLock<Vec<LazyItem<u32>>>>; 64],
    pub lazy_children: LazyItemArray<InvertedIndexSparseAnnNodeBasic, 16>,
}

impl InvertedIndexSparseAnnNodeBasic {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        let data: [Arc<RwLock<Vec<LazyItem<u32>>>>; 64] =
            from_fn(|_| Arc::new(RwLock::new(Vec::new())));

        InvertedIndexSparseAnnNodeBasic {
            dim_index,
            implicit,
            data,
            lazy_children: LazyItemArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: ArcShift<InvertedIndexSparseAnnNodeBasic>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> ArcShift<InvertedIndexSparseAnnNodeBasic> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(
                0.into(),
                0u16,
                InvertedIndexSparseAnnNodeBasic::new(new_dim_index, true),
            );
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
                {
                    let res: Arc<InvertedIndexSparseAnnNodeBasic> = child.get_data(cache.clone());
                    current_node = ArcShift::new((*res).clone());
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
    pub fn insert(node: ArcShift<InvertedIndexSparseAnnNodeBasic>, value: f32, vector_id: u32) {
        let quantized_value = Self::quantize(value);
        let data = node.data.clone();

        // Insert into the specific LazyItem at the index quantized_value
        if let Some(arc_lazy_item) = data.get(quantized_value as usize) {
            let mut vec = arc_lazy_item.write().unwrap();
            vec.push(LazyItem::new(0.into(), 0u16, vector_id));
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
                for (index, arc_rwlock_lazy_item) in self.data.iter().enumerate() {
                    let arc_rwlock_lazy_item = arc_rwlock_lazy_item.read().unwrap();
                    if arc_rwlock_lazy_item
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

/// [InvertedIndexSparseAnnBasic] is a improved version which only holds quantized u8 values instead of f32 inside [InvertedIndexSparseAnnNodeBasic]
#[derive(Clone)]
pub struct InvertedIndexSparseAnnBasic {
    pub root: ArcShift<InvertedIndexSparseAnnNodeBasic>,
    pub cache: Arc<NodeRegistry>,
}

impl InvertedIndexSparseAnnBasic {
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            1.0,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndexSparseAnnBasic {
            root: ArcShift::new(InvertedIndexSparseAnnNodeBasic::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<ArcShift<InvertedIndexSparseAnnNodeBasic>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            let node_res = child.get_data(self.cache.clone());
            current_node = ArcShift::new((*node_res).clone());
        }

        Some(current_node)
    }

    //Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNodeBasic::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        //value will be quantized while being inserted into the Node.
        InvertedIndexSparseAnnNodeBasic::insert(node, value, vector_id)
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

#[derive(Clone)]
pub struct InvertedIndexSparseAnnNodeBasicTSHashmap {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: TSHashTable<u8, Pagepool<PAGE_SIZE>>,
    pub lazy_children: LazyItemArray<InvertedIndexSparseAnnNodeBasicTSHashmap, 16>,
    pub cuckoo_filter_tree: Arc<CuckooFilterTreeNode>,
}

#[derive(Clone)]
pub struct InvertedIndexSparseAnnBasicTSHashmap {
    pub root: ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>,
    pub cache: Arc<NodeRegistry>,
}

impl InvertedIndexSparseAnnNodeBasicTSHashmap {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        let data = TSHashTable::new(16);

        InvertedIndexSparseAnnNodeBasicTSHashmap {
            dim_index,
            implicit,
            data,
            lazy_children: LazyItemArray::new(),
            cuckoo_filter_tree: Arc::new(CuckooFilterTreeNode::build_tree(5, 0, 0.0, 10.0)),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_or_create_node(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(
                0.into(),
                0u16,
                InvertedIndexSparseAnnNodeBasicTSHashmap::new(new_dim_index, true),
            );
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
                {
                    let res: Arc<InvertedIndexSparseAnnNodeBasicTSHashmap> =
                        child.get_data(cache.clone());
                    current_node = ArcShift::new((*res).clone());
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
    pub fn insert(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicTSHashmap>,
        value: f32,
        vector_id: u32,
    ) {
        let quantized_value = Self::quantize(value);
        let data = node.data.clone();
        data.get_or_create(quantized_value, || Pagepool::default());
        data.mutate(quantized_value, |x| {
            let mut vecof_vec_id = x.unwrap();
            vecof_vec_id.push(vector_id);
            Some(vecof_vec_id)
        });
        println!("vector_id -> {vector_id}");
        let mut tree = node.cuckoo_filter_tree.clone();
        Arc::make_mut(&mut tree).add_item(vector_id as u64, value);
        println!("vector_id_2 -> {vector_id}");
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
                let res = self.data.to_list();
                for (x, y) in res {
                    if y.contains(vector_id) {
                        return Some(x);
                    }
                }
                None
            }
        }
    }
}

impl InvertedIndexSparseAnnBasicTSHashmap {
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            1.0,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndexSparseAnnBasicTSHashmap {
            root: ArcShift::new(InvertedIndexSparseAnnNodeBasicTSHashmap::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
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

    //Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNodeBasicTSHashmap::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        //value will be quantized while being inserted into the Node.
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
}

#[derive(Clone)]
pub struct InvertedIndexSparseAnnNodeBasicDashMap {
    pub dim_index: u32,
    pub implicit: bool,
    pub data: DashMap<u32, u8>,
    pub lazy_children: LazyItemArray<InvertedIndexSparseAnnNodeBasicDashMap, 16>,
}

#[derive(Clone)]
pub struct InvertedIndexSparseAnnBasicDashMap {
    pub root: ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>,
    pub cache: Arc<NodeRegistry>,
}

impl InvertedIndexSparseAnnNodeBasicDashMap {
    pub fn new(dim_index: u32, implicit: bool) -> Self {
        let data: DashMap<u32, u8> = DashMap::new();

        InvertedIndexSparseAnnNodeBasicDashMap {
            dim_index,
            implicit,
            data,
            lazy_children: LazyItemArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    fn find_or_create_node(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>,
        path: &[usize],
        cache: Arc<NodeRegistry>,
    ) -> ArcShift<InvertedIndexSparseAnnNodeBasicDashMap> {
        let mut current_node = node;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            let new_child = LazyItem::new(
                0.into(),
                0u16,
                InvertedIndexSparseAnnNodeBasicDashMap::new(new_dim_index, true),
            );
            loop {
                if let Some(child) = current_node
                    .lazy_children
                    .checked_insert(child_index, new_child.clone())
                {
                    let res: Arc<InvertedIndexSparseAnnNodeBasicDashMap> =
                        child.get_data(cache.clone());
                    current_node = ArcShift::new((*res).clone());
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
    pub fn insert(
        node: ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>,
        value: f32,
        vector_id: u32,
    ) {
        let quantized_value = Self::quantize(value);
        let data = node.data.clone();
        data.insert(vector_id, quantized_value);
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
                let res = self.data.get(&vector_id);
                match res {
                    Some(val) => {
                        let p = *val;
                        return Some(p);
                    }
                    None => return None,
                }
            }
        }
    }
}

impl InvertedIndexSparseAnnBasicDashMap {
    pub fn new() -> Self {
        let bufmans = Arc::new(BufferManagerFactory::new(
            Path::new(".").into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            1.0,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        InvertedIndexSparseAnnBasicDashMap {
            root: ArcShift::new(InvertedIndexSparseAnnNodeBasicDashMap::new(0, false)),
            cache,
        }
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(
        &self,
        dim_index: u32,
    ) -> Option<ArcShift<InvertedIndexSparseAnnNodeBasicDashMap>> {
        let mut current_node = self.root.clone();
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.lazy_children.get(child_index)?;
            let node_res = child.get_data(self.cache.clone());
            current_node = ArcShift::new((*node_res).clone());
        }

        Some(current_node)
    }

    //Fetches quantized u8 value for a dim_index and vector_Id present at respective node in index
    pub fn get(&self, dim_index: u32, vector_id: u32) -> Option<u8> {
        self.root
            .shared_get()
            .get(dim_index, vector_id, self.cache.clone())
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(&self, dim_index: u32, value: f32, vector_id: u32) {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = InvertedIndexSparseAnnNodeBasicDashMap::find_or_create_node(
            self.root.clone(),
            &path,
            self.cache.clone(),
        );
        //value will be quantized while being inserted into the Node.
        InvertedIndexSparseAnnNodeBasicDashMap::insert(node, value, vector_id)
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
