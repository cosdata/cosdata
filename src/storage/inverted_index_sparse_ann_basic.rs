use arcshift::ArcShift;
use core::array::from_fn;
use dashmap::DashMap;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::{path::Path, sync::RwLock};

use std::sync::Arc;

use crate::models::atomic_array::AtomicArray;
use crate::models::buffered_io::{BufIoError, BufferManager};
use crate::models::cache_loader::InvertedIndexCache;
use crate::models::fixedset::VersionedInvertedFixedSetIndex;
use crate::models::prob_lazy_load::lazy_item::ProbLazyItem;
use crate::models::serializer::inverted::InvertedIndexSerialize;
use crate::models::types::FileOffset;
use crate::models::versioning::Hash;
use crate::models::{
    buffered_io::BufferManagerFactory,
    cache_loader::NodeRegistry,
    common::TSHashTable,
    lazy_load::{LazyItem, LazyItemArray},
    types::SparseVector,
};

use super::page::VersionedPagepool;

// Size of a page in the hash table
pub const PAGE_SIZE: usize = 32;
pub const FIXED_SET_SIZE: usize = 8;

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
            8192,
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

#[cfg_attr(test, derive(PartialEq, Debug))]
pub struct InvertedIndexSparseAnnNodeBasicTSHashmapData {
    pub map: TSHashTable<u8, VersionedPagepool<PAGE_SIZE>>,
    pub max_key: u8,
}

impl InvertedIndexSparseAnnNodeBasicTSHashmapData {
    pub fn new(quantization_bits: u8) -> Self {
        Self {
            map: TSHashTable::new(16),
            max_key: ((1u32 << quantization_bits) - 1) as u8,
        }
    }
}

// #[derive(Debug)]
pub struct InvertedIndexSparseAnnNodeBasicTSHashmap {
    pub is_serialized: AtomicBool,
    pub is_dirty: AtomicBool,
    pub file_offset: FileOffset,
    pub dim_index: u32,
    pub implicit: bool,
    // (4, 5, 6)
    pub quantization_bits: u8,
    pub data: *mut ProbLazyItem<InvertedIndexSparseAnnNodeBasicTSHashmapData>,
    pub children: AtomicArray<InvertedIndexSparseAnnNodeBasicTSHashmap, 16>,
    pub fixed_sets: *mut ProbLazyItem<VersionedInvertedFixedSetIndex>,
}

#[cfg(test)]
impl PartialEq for InvertedIndexSparseAnnNodeBasicTSHashmap {
    fn eq(&self, other: &Self) -> bool {
        self.file_offset == other.file_offset
            && self.dim_index == other.dim_index
            && self.implicit == other.implicit
            && self.quantization_bits == other.quantization_bits
            && unsafe { *self.data == *other.data }
            && self.children == other.children
            && unsafe { *self.fixed_sets == *other.fixed_sets }
    }
}

#[cfg(test)]
impl std::fmt::Debug for InvertedIndexSparseAnnNodeBasicTSHashmap {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("InvertedIndexSparseAnnNodeBasicTSHashmap")
            .field("file_offset", &self.file_offset)
            .field("dim_index", &self.dim_index)
            .field("implicit", &self.implicit)
            .field("quantization_bits", &self.quantization_bits)
            .field("data", unsafe { &*self.data })
            .field("children", &self.children)
            .field("fixed_sets", unsafe { &*self.fixed_sets })
            .finish()
    }
}

pub struct InvertedIndexSparseAnnBasicTSHashmap {
    pub root: Arc<InvertedIndexSparseAnnNodeBasicTSHashmap>,
    pub cache: Arc<InvertedIndexCache>,
    pub data_file_parts: u8,
    pub offset_counter: AtomicU32,
    pub node_size: u32,
}

#[cfg(test)]
impl PartialEq for InvertedIndexSparseAnnBasicTSHashmap {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
            && self.offset_counter.load(Ordering::Relaxed)
                == other.offset_counter.load(Ordering::Relaxed)
            && self.node_size == other.node_size
    }
}

#[cfg(test)]
impl std::fmt::Debug for InvertedIndexSparseAnnBasicTSHashmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexSparseAnnBasicTSHashmap")
            .field("root", &self.root)
            .field(
                "offset_counter",
                &self.offset_counter.load(Ordering::Relaxed),
            )
            .field("node_size", &self.node_size)
            .finish()
    }
}

unsafe impl Send for InvertedIndexSparseAnnNodeBasicTSHashmap {}
unsafe impl Sync for InvertedIndexSparseAnnNodeBasicTSHashmap {}
unsafe impl Send for InvertedIndexSparseAnnBasicTSHashmap {}
unsafe impl Sync for InvertedIndexSparseAnnBasicTSHashmap {}

impl InvertedIndexSparseAnnNodeBasicTSHashmap {
    pub fn new(
        dim_index: u32,
        implicit: bool,
        // 4, 5, 6
        quantization_bits: u8,
        version_id: Hash,
        file_offset: FileOffset,
    ) -> Self {
        let data = ProbLazyItem::new(
            InvertedIndexSparseAnnNodeBasicTSHashmapData::new(quantization_bits),
            0.into(),
            0,
            false,
            FileOffset(file_offset.0 + 5),
        );

        let fixed_sets = ProbLazyItem::new(
            VersionedInvertedFixedSetIndex::new(quantization_bits, version_id),
            0.into(),
            0,
            false,
            FileOffset(file_offset.0 + (1u32 << quantization_bits) * 4 + 69),
        );

        Self {
            is_serialized: AtomicBool::new(false),
            is_dirty: AtomicBool::new(true),
            file_offset,
            dim_index,
            implicit,
            data,
            children: AtomicArray::new(),
            quantization_bits,
            fixed_sets,
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_or_create_node(
        &self,
        path: &[usize],
        version_id: Hash,
        mut offset_fn: impl FnMut() -> u32,
    ) -> &Self {
        let mut current_node = self;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + POWERS_OF_4[child_index];
            if let Some(child) = current_node.children.get(child_index) {
                let res = unsafe { &*child };
                current_node = res;
                continue;
            }
            let (new_child, is_newly_created) =
                current_node.children.get_or_insert(child_index, || {
                    Box::into_raw(Box::new(Self::new(
                        new_dim_index,
                        true,
                        self.quantization_bits,
                        version_id,
                        FileOffset(offset_fn()),
                    )))
                });
            if is_newly_created {
                self.is_dirty.store(true, Ordering::Release);
            }
            let res = unsafe { &*new_child };
            current_node = res;
        }

        current_node
    }

    pub fn quantize(&self, value: f32, values_upper_bound: f32) -> u8 {
        let quantization = ((1u32 << self.quantization_bits) - 1) as u8;
        let max_val = quantization as f32;
        (((value / values_upper_bound) * max_val).clamp(0.0, max_val) as u8).min(quantization)
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Finds the quantized value and pushes the vec_Id in array at index = quantized_value
    pub fn insert(
        &self,
        value: f32,
        vector_id: u32,
        cache: &InvertedIndexCache,
        version: Hash,
        values_upper_bound: f32,
    ) -> Result<(), BufIoError> {
        let quantized_value = self.quantize(value, values_upper_bound);
        unsafe { &*self.data }
            .try_get_data(cache, self.dim_index)?
            .map
            .modify_or_insert(
                quantized_value,
                |list| {
                    list.push(version, vector_id);
                },
                || {
                    let mut pool = VersionedPagepool::new(version);
                    pool.push(version, vector_id);
                    pool
                },
            );
        let sets = unsafe { &*self.fixed_sets }.try_get_data(cache, self.dim_index)?;
        sets.insert(version, quantized_value, vector_id);
        self.is_dirty.store(true, Ordering::Release);
        Ok(())
    }

    pub fn find_key_of_id(
        &self,
        vector_id: u32,
        cache: &InvertedIndexCache,
    ) -> Result<Option<u8>, BufIoError> {
        Ok(unsafe { &*self.fixed_sets }
            .try_get_data(cache, self.dim_index)?
            .search(vector_id))
    }

    /// See [`crate::models::serializer::inverted::node`] for how its calculated
    pub fn get_serialized_size(quantization_bits: u8) -> u32 {
        let qv = 1u32 << quantization_bits;

        qv * 4 + 73
    }
}

impl InvertedIndexSparseAnnBasicTSHashmap {
    pub fn new(
        root_path: PathBuf,
        quantization_bits: u8,
        version: Hash,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(root_path.join("index-tree.dim"))?;
        let node_size =
            InvertedIndexSparseAnnNodeBasicTSHashmap::get_serialized_size(quantization_bits);
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(node_size);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.into(),
            |root, idx: &u8| root.join(format!("{}.idat", idx)),
            8192,
        ));
        let cache = Arc::new(InvertedIndexCache::new(
            dim_bufman,
            data_bufmans,
            data_file_parts,
        ));

        Ok(InvertedIndexSparseAnnBasicTSHashmap {
            root: Arc::new(InvertedIndexSparseAnnNodeBasicTSHashmap::new(
                0,
                false,
                quantization_bits,
                version,
                FileOffset(0),
            )),
            cache,
            data_file_parts,
            offset_counter,
            node_size,
        })
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<&InvertedIndexSparseAnnNodeBasicTSHashmap> {
        let mut current_node = &*self.root;
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.children.get(child_index)?;
            let node_res = unsafe { &*child };
            current_node = node_res;
        }

        Some(current_node)
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(
        &self,
        dim_index: u32,
        value: f32,
        vector_id: u32,
        version: Hash,
        values_upper_bound: f32,
    ) -> Result<(), BufIoError> {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = self.root.find_or_create_node(&path, version, || {
            self.offset_counter
                .fetch_add(self.node_size, Ordering::Relaxed)
        });
        //value will be quantized while being inserted into the Node.
        node.insert(value, vector_id, &self.cache, version, values_upper_bound)
    }

    /// Adds a sparse vector to the index.
    pub fn add_sparse_vector(
        &self,
        vector: SparseVector,
        version: Hash,
        values_upper_bound: f32,
    ) -> Result<(), BufIoError> {
        let vector_id = vector.vector_id;
        vector
            .entries
            .par_iter()
            .map(|(dim_index, value)| {
                if *value != 0.0 {
                    return self.insert(*dim_index, *value, vector_id, version, values_upper_bound);
                }
                Ok(())
            })
            .collect()
    }

    pub fn serialize(&self) -> Result<(), BufIoError> {
        let cursor = self.cache.dim_bufman.open_cursor()?;
        self.root.serialize(
            &self.cache.dim_bufman,
            &self.cache.data_bufmans,
            0,
            self.data_file_parts,
            cursor,
        )?;
        self.cache.dim_bufman.close_cursor(cursor)?;
        Ok(())
    }

    pub fn deserialize(
        root_path: PathBuf,
        quantization_bits: u8,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(root_path.join("index-tree.dim"))?;
        let node_size =
            InvertedIndexSparseAnnNodeBasicTSHashmap::get_serialized_size(quantization_bits);
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(dim_bufman.file_size() as u32);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.into(),
            |root, idx: &u8| root.join(format!("{}.idat", idx)),
            8192,
        ));
        let cache = Arc::new(InvertedIndexCache::new(
            dim_bufman,
            data_bufmans,
            data_file_parts,
        ));

        Ok(Self {
            root: Arc::new(InvertedIndexSparseAnnNodeBasicTSHashmap::deserialize(
                &cache.dim_bufman,
                &cache.data_bufmans,
                FileOffset(0),
                0,
                data_file_parts,
                &cache,
            )?),
            cache,
            data_file_parts,
            offset_counter,
            node_size,
        })
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
            8192,
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
