use rayon::prelude::*;
use std::{
    fs::OpenOptions,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
};

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    common::TSHashTable,
    lazy_item::LazyItem,
    serializer::inverted::InvertedIndexSerialize,
    types::{FileOffset, SparseVector},
    utils::calculate_path,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

#[cfg_attr(test, derive(PartialEq, Debug))]
pub struct InvertedIndexNodeData {
    pub map: TSHashTable<u8, VersionedVec<u32>>,
    pub max_key: u8,
}

impl InvertedIndexNodeData {
    pub fn new(quantization_bits: u8) -> Self {
        Self {
            map: TSHashTable::new(16),
            max_key: ((1u32 << quantization_bits) - 1) as u8,
        }
    }
}

pub struct InvertedIndexNode {
    pub is_serialized: AtomicBool,
    pub is_dirty: AtomicBool,
    pub file_offset: FileOffset,
    pub dim_index: u32,
    pub implicit: bool,
    // (4, 5, 6)
    pub quantization_bits: u8,
    pub data: *mut LazyItem<InvertedIndexNodeData, ()>,
    pub children: AtomicArray<InvertedIndexNode, 16>,
}

#[cfg(test)]
impl PartialEq for InvertedIndexNode {
    fn eq(&self, other: &Self) -> bool {
        self.file_offset == other.file_offset
            && self.dim_index == other.dim_index
            && self.implicit == other.implicit
            && self.quantization_bits == other.quantization_bits
            && unsafe { *self.data == *other.data }
            && self.children == other.children
    }
}

#[cfg(test)]
impl std::fmt::Debug for InvertedIndexNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("InvertedIndexNode")
            .field("file_offset", &self.file_offset)
            .field("dim_index", &self.dim_index)
            .field("implicit", &self.implicit)
            .field("quantization_bits", &self.quantization_bits)
            .field("data", unsafe { &*self.data })
            .field("children", &self.children)
            .finish()
    }
}

pub struct InvertedIndexRoot {
    pub root: InvertedIndexNode,
    pub cache: InvertedIndexCache,
    pub offset_counter: AtomicU32,
    pub node_size: u32,
}

#[cfg(test)]
impl PartialEq for InvertedIndexRoot {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
            && self.offset_counter.load(Ordering::Relaxed)
                == other.offset_counter.load(Ordering::Relaxed)
            && self.node_size == other.node_size
    }
}

#[cfg(test)]
impl std::fmt::Debug for InvertedIndexRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexRoot")
            .field("root", &self.root)
            .field(
                "offset_counter",
                &self.offset_counter.load(Ordering::Relaxed),
            )
            .field("node_size", &self.node_size)
            .finish()
    }
}

unsafe impl Send for InvertedIndexNode {}
unsafe impl Sync for InvertedIndexNode {}
unsafe impl Send for InvertedIndexRoot {}
unsafe impl Sync for InvertedIndexRoot {}

impl InvertedIndexNode {
    pub fn new(
        dim_index: u32,
        implicit: bool,
        // 4, 5, 6
        quantization_bits: u8,
        file_offset: FileOffset,
    ) -> Self {
        let data = LazyItem::new(
            InvertedIndexNodeData::new(quantization_bits),
            (),
            FileOffset(file_offset.0 + 5),
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
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_or_create_node(&self, path: &[u8], mut offset_fn: impl FnMut() -> u32) -> &Self {
        let mut current_node = self;
        for &child_index in path {
            let new_dim_index = current_node.dim_index + (1u32 << (child_index * 2));
            if let Some(child) = current_node.children.get(child_index as usize) {
                let res = unsafe { &*child };
                current_node = res;
                continue;
            }
            let (new_child, _is_newly_created) =
                current_node
                    .children
                    .get_or_insert(child_index as usize, || {
                        Box::into_raw(Box::new(Self::new(
                            new_dim_index,
                            true,
                            self.quantization_bits,
                            FileOffset(offset_fn()),
                        )))
                    });
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
        version: VersionNumber,
        values_upper_bound: f32,
    ) -> Result<(), BufIoError> {
        let quantized_value = self.quantize(value, values_upper_bound);
        unsafe { &*self.data }
            .try_get_data(cache)?
            .map
            .modify_or_insert(
                quantized_value,
                |list| {
                    list.push(version, vector_id);
                },
                || {
                    let mut pool = VersionedVec::new(version);
                    pool.push(version, vector_id);
                    pool
                },
            );
        self.is_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Delete a value from the index at the specified dimension index.
    /// Finds the quantized value and (soft) deletes the vec_id
    pub fn delete(
        &self,
        value: f32,
        vector_id: u32,
        cache: &InvertedIndexCache,
        version: VersionNumber,
        values_upper_bound: f32,
    ) -> Result<(), BufIoError> {
        let quantized_value = self.quantize(value, values_upper_bound);
        unsafe { &*self.data }
            .try_get_data(cache)?
            .map
            .with_value_mut(&quantized_value, |list| {
                list.delete(version, vector_id);
            });
        self.is_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// See [`crate::models::serializer::inverted::node`] for how its calculated
    pub fn get_serialized_size(quantization_bits: u8) -> u32 {
        let qv = 1u32 << quantization_bits;

        qv * 8 + 69
    }
}

impl InvertedIndexRoot {
    pub fn new(root_path: PathBuf, quantization_bits: u8) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root_path.join("index-tree.dim"))?;
        let node_size = InvertedIndexNode::get_serialized_size(quantization_bits);
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(node_size);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.into(),
            |root, version: &VersionNumber| root.join(format!("{}.idat", **version)),
            8192,
        ));
        let cache = InvertedIndexCache::new(dim_bufman, data_bufmans);

        Ok(InvertedIndexRoot {
            root: InvertedIndexNode::new(0, false, quantization_bits, FileOffset(0)),
            cache,
            offset_counter,
            node_size,
        })
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<&InvertedIndexNode> {
        let mut current_node = &self.root;
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.children.get(child_index as usize)?;
            let node_res = unsafe { &*child };
            current_node = node_res;
        }

        Some(current_node)
    }

    // Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(
        &self,
        dim_index: u32,
        value: f32,
        vector_id: u32,
        version: VersionNumber,
        values_upper_bound: f32,
    ) -> Result<(), BufIoError> {
        let path = calculate_path(dim_index, self.root.dim_index);
        let node = self.root.find_or_create_node(&path, || {
            self.offset_counter
                .fetch_add(self.node_size, Ordering::Relaxed)
        });
        assert_eq!(node.dim_index, dim_index);
        // value will be quantized while being inserted into the Node.
        node.insert(value, vector_id, &self.cache, version, values_upper_bound)
    }

    pub fn delete(
        &self,
        dim_index: u32,
        value: f32,
        vector_id: u32,
        version: VersionNumber,
        values_upper_bound: f32,
    ) -> Result<(), BufIoError> {
        let node_optional = self.find_node(dim_index);
        let Some(node) = node_optional else {
            return Ok(());
        };
        assert_eq!(node.dim_index, dim_index);
        // value will be quantized while being deleted from the Node.
        node.delete(value, vector_id, &self.cache, version, values_upper_bound)
    }

    /// Adds a sparse vector to the index.
    #[allow(unused)]
    pub fn add_sparse_vector(
        &self,
        vector: SparseVector,
        version: VersionNumber,
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
        self.root
            .serialize(&self.cache.dim_bufman, &self.cache.data_bufmans, cursor)?;
        self.cache.dim_bufman.close_cursor(cursor)?;
        Ok(())
    }

    pub fn deserialize(root_path: PathBuf, quantization_bits: u8) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root_path.join("index-tree.dim"))?;
        let node_size = InvertedIndexNode::get_serialized_size(quantization_bits);
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(dim_bufman.file_size() as u32);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.into(),
            |root, version: &VersionNumber| root.join(format!("{}.idat", **version)),
            8192,
        ));
        let cache = InvertedIndexCache::new(dim_bufman, data_bufmans);

        Ok(Self {
            root: InvertedIndexNode::deserialize(
                &cache.dim_bufman,
                &cache.data_bufmans,
                FileOffset(0),
                VersionNumber::from(u32::MAX), // not used
                &cache,
            )?,
            cache,
            offset_counter,
            node_size,
        })
    }
}
