use std::{
    fs::OpenOptions,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering},
        Arc, RwLock,
    },
};

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexIDFCache,
    common::TSHashTable,
    page::VersionedPagepool,
    prob_lazy_load::lazy_item::ProbLazyItem,
    serializer::inverted_idf::{InvertedIndexIDFSerialize, INVERTED_INDEX_DATA_CHUNK_SIZE},
    types::FileOffset,
    utils::calculate_path,
    versioning::Hash,
};

// Size of a page in the hash table
pub const PAGE_SIZE: usize = 32;

// Term quotient (upper 16 bits of the hash)
pub type TermQuotient = u16;
// Quantized term frequency value
pub type QuantizedFrequency = u8;
// Collection of Document IDs with their term frequencies
pub type DocumentIDList = VersionedPagepool<PAGE_SIZE>;
// Inner map from quantized frequencies to documents
pub type TermFrequencyMap = TSHashTable<QuantizedFrequency, DocumentIDList>;
// Outer map from term quotients to TermInfo
pub type QuotientMap = TSHashTable<TermQuotient, Arc<TermInfo>>;

pub struct TermInfo {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub frequency_map: TermFrequencyMap,
    pub sequence_idx: u16,
    pub documents_count: AtomicU32,
}

impl TermInfo {
    #[allow(unused)]
    pub fn new(sequence_idx: u16) -> Self {
        Self {
            serialized_at: RwLock::new(None),
            frequency_map: TSHashTable::new(16),
            sequence_idx,
            documents_count: AtomicU32::new(0),
        }
    }
}

#[cfg(test)]
impl PartialEq for TermInfo {
    fn eq(&self, other: &Self) -> bool {
        *self.serialized_at.read().unwrap() == *other.serialized_at.read().unwrap()
            && self.frequency_map == other.frequency_map
            && self.sequence_idx == other.sequence_idx
            && self.documents_count.load(Ordering::Relaxed)
                == other.documents_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl std::fmt::Debug for TermInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermInfo")
            .field("serialized_at", &*self.serialized_at.read().unwrap())
            .field("frequency_map", &self.frequency_map)
            .field("sequence_idx", &self.sequence_idx)
            .field(
                "documents_count",
                &self.documents_count.load(Ordering::Relaxed),
            )
            .finish()
    }
}

pub struct InvertedIndexIDFNodeData {
    // Map from term quotients to TermInfo
    pub map: QuotientMap,
    pub map_len: AtomicU16,
    pub num_entries_serialized: RwLock<u16>,
}

#[cfg(test)]
impl PartialEq for InvertedIndexIDFNodeData {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
            && self.map_len.load(Ordering::Relaxed) == other.map_len.load(Ordering::Relaxed)
            && *self.num_entries_serialized.read().unwrap()
                == *other.num_entries_serialized.read().unwrap()
    }
}

#[cfg(test)]
impl std::fmt::Debug for InvertedIndexIDFNodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexIDFNodeData")
            .field("map", &self.map)
            .field("map_len", &self.map_len.load(Ordering::Relaxed))
            .field(
                "num_entries_serialized",
                &*self.num_entries_serialized.read().unwrap(),
            )
            .finish()
    }
}

impl Default for InvertedIndexIDFNodeData {
    fn default() -> Self {
        Self {
            map: QuotientMap::new(16),
            map_len: AtomicU16::new(0),
            num_entries_serialized: RwLock::new(0),
        }
    }
}

impl InvertedIndexIDFNodeData {
    pub fn new() -> Self {
        Self::default()
    }

    // Get IDF score for a term (represented by quotient)
    pub fn get_idf(&self, quotient: TermQuotient, global_document_count: u32) -> f32 {
        let doc_freq = self
            .map
            .lookup(&quotient)
            .map(|v| v.documents_count.load(Ordering::Relaxed))
            .unwrap_or(0);

        // BM25 probabilistic IDF formula
        (((global_document_count - doc_freq) as f32 + 0.5) / (doc_freq as f32 + 0.5)).ln() + 1.0
    }
}

pub struct InvertedIndexIDFNode {
    pub is_serialized: AtomicBool,
    pub quantization_bits: u8,
    pub is_dirty: AtomicBool,
    pub file_offset: FileOffset,
    pub dim_index: u32,
    pub implicit: bool,
    pub data: *mut ProbLazyItem<InvertedIndexIDFNodeData>,
    pub children: AtomicArray<InvertedIndexIDFNode, 16>,
}

#[cfg(test)]
impl PartialEq for InvertedIndexIDFNode {
    fn eq(&self, other: &Self) -> bool {
        self.file_offset == other.file_offset
            && self.dim_index == other.dim_index
            && self.quantization_bits == other.quantization_bits
            && self.implicit == other.implicit
            && self.children == other.children
            && unsafe { *self.data == *other.data }
    }
}

#[cfg(test)]
impl std::fmt::Debug for InvertedIndexIDFNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("InvertedIndexIDFNode")
            .field("file_offset", &self.file_offset)
            .field("quantization_bits", &self.quantization_bits)
            .field("dim_index", &self.dim_index)
            .field("implicit", &self.implicit)
            .field("children", &self.children)
            .finish()
    }
}

pub struct InvertedIndexIDFRoot {
    pub root: InvertedIndexIDFNode,
    pub cache: InvertedIndexIDFCache,
    // total number of documents in the index
    pub total_documents_count: AtomicU32,
    pub data_file_parts: u8,
}

#[cfg(test)]
impl PartialEq for InvertedIndexIDFRoot {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
            && self.total_documents_count.load(Ordering::Relaxed)
                == other.total_documents_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl std::fmt::Debug for InvertedIndexIDFRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexIDFRoot")
            .field("root", &self.root)
            .field(
                "total_documents_count",
                &self.total_documents_count.load(Ordering::Relaxed),
            )
            .field("data_file_parts", &self.data_file_parts)
            .finish()
    }
}

unsafe impl Send for InvertedIndexIDFNode {}
unsafe impl Sync for InvertedIndexIDFNode {}
unsafe impl Send for InvertedIndexIDFRoot {}
unsafe impl Sync for InvertedIndexIDFRoot {}

impl InvertedIndexIDFNode {
    pub fn new(
        dim_index: u32,
        implicit: bool,
        quantization_bits: u8,
        file_offset: FileOffset,
    ) -> Self {
        let data = ProbLazyItem::new(
            InvertedIndexIDFNodeData::new(),
            0.into(),
            0,
            false,
            FileOffset(file_offset.0 + 5),
        );

        Self {
            is_serialized: AtomicBool::new(false),
            quantization_bits,
            is_dirty: AtomicBool::new(true),
            file_offset,
            dim_index,
            implicit,
            data,
            children: AtomicArray::new(),
        }
    }

    /// Finds or creates the node where the data should be inserted.
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_or_create_node(&self, path: &[u8], mut offset_fn: impl FnMut() -> u32) -> &Self {
        let mut current_node = self;
        for &child_index in path {
            let new_dim_index = (current_node.dim_index + 1u32) << (child_index * 2);
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

    fn quantize(&self, value: f32) -> u8 {
        ((value / 2.0) * (1u32 << self.quantization_bits) as f32)
            .min(((1u32 << self.quantization_bits) - 1) as f32) as u8
    }

    /// Inserts a value into the index at the specified dimension index.
    /// Finds the quantized value and pushes the vec_Id in array at index = quantized_value
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &self,
        quotient: TermQuotient,
        value: f32,
        document_id: u32,
        cache: &InvertedIndexIDFCache,
        version: Hash,
    ) -> Result<(), BufIoError> {
        // Get node data
        let data = unsafe { &*self.data }.try_get_data(cache, self.dim_index)?;

        let quantized_value = self.quantize(value);

        // Get or create inner map for this quotient
        data.map.modify_or_insert(
            quotient,
            |v| {
                // Update or insert quantized value in inner map
                v.frequency_map.modify_or_insert(
                    quantized_value,
                    |document_list| {
                        document_list.push(version, document_id);
                    },
                    || {
                        let pool = DocumentIDList::new(version);
                        pool.push(version, document_id);
                        pool
                    },
                );

                v.documents_count.fetch_add(1, Ordering::Relaxed);
            },
            || {
                // Create new inner map if quotient not found
                let frequency_map = TermFrequencyMap::new(8);
                let sequence_idx = data.map_len.fetch_add(1, Ordering::Relaxed);
                let pool = DocumentIDList::new(version);
                pool.push(version, document_id);
                frequency_map.insert(quantized_value, pool);
                Arc::new(TermInfo {
                    serialized_at: RwLock::new(None),
                    frequency_map,
                    sequence_idx,
                    documents_count: AtomicU32::new(0),
                })
            },
        );

        // Mark node as dirty
        self.is_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// See [`crate::models::serializer::inverted_idf::node`] for how its calculated
    pub fn get_serialized_size() -> u32 {
        INVERTED_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 75
    }
}

impl InvertedIndexIDFRoot {
    pub fn new(
        root_path: PathBuf,
        quantization_bits: u8,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let dim_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root_path.join("index-tree.dim"))?;
        let node_size = InvertedIndexIDFNode::get_serialized_size();
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(node_size + 4);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.clone().into(),
            |root, idx: &u8| root.join(format!("{}.idat", idx)),
            8192,
        ));
        let cache = InvertedIndexIDFCache::new(
            dim_bufman,
            data_bufmans,
            offset_counter,
            data_file_parts,
            quantization_bits,
        );

        Ok(InvertedIndexIDFRoot {
            root: InvertedIndexIDFNode::new(0, false, quantization_bits, FileOffset(4)),
            cache,
            total_documents_count: AtomicU32::new(0),
            data_file_parts,
        })
    }

    /// Finds the node at a given dimension
    /// Traverses the tree iteratively and returns a reference to the node.
    pub fn find_node(&self, dim_index: u32) -> Option<&InvertedIndexIDFNode> {
        let mut current_node = &self.root;
        let path = calculate_path(dim_index, self.root.dim_index);
        for child_index in path {
            let child = current_node.children.get(child_index as usize)?;
            let node_res = unsafe { &*child };
            current_node = node_res;
        }

        Some(current_node)
    }

    //Inserts vec_id, quantized value u8 at particular node based on path
    pub fn insert(
        &self,
        hash_dim: u32,
        value: f32,
        document_id: u32,
        version: Hash,
    ) -> Result<(), BufIoError> {
        // Split the hash dimension
        let storage_dim = hash_dim % 65536;
        let quotient = (hash_dim / 65536) as TermQuotient;

        let path = calculate_path(storage_dim, self.root.dim_index);
        let node = self.root.find_or_create_node(&path, || {
            self.cache.offset_counter.fetch_add(
                InvertedIndexIDFNode::get_serialized_size(),
                Ordering::Relaxed,
            )
        });
        // value will be quantized while being inserted into the Node.
        node.insert(quotient, value, document_id, &self.cache, version)
    }

    pub fn serialize(&self) -> Result<(), BufIoError> {
        let cursor = self.cache.dim_bufman.open_cursor()?;
        self.cache
            .dim_bufman
            .update_u32_with_cursor(cursor, self.total_documents_count.load(Ordering::Relaxed))?;
        self.root.serialize(
            &self.cache.dim_bufman,
            &self.cache.data_bufmans,
            &self.cache.offset_counter,
            self.root.quantization_bits,
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
            .truncate(false)
            .open(root_path.join("index-tree.dim"))?;
        let node_size = InvertedIndexIDFNode::get_serialized_size();
        let dim_bufman = Arc::new(BufferManager::new(dim_file, node_size as usize * 1000)?);
        let offset_counter = AtomicU32::new(dim_bufman.file_size() as u32);
        let data_bufmans = Arc::new(BufferManagerFactory::new(
            root_path.clone().into(),
            |root, idx: &u8| root.join(format!("{}.idat", idx)),
            8192,
        ));
        let cache = InvertedIndexIDFCache::new(
            dim_bufman,
            data_bufmans,
            offset_counter,
            data_file_parts,
            quantization_bits,
        );
        let root = InvertedIndexIDFNode::deserialize(
            &cache.dim_bufman,
            &cache.data_bufmans,
            FileOffset(4),
            quantization_bits,
            0,
            data_file_parts,
            &cache,
        )?;
        let cursor = cache.dim_bufman.open_cursor()?;
        let total_documents_count = AtomicU32::new(cache.dim_bufman.read_u32_with_cursor(cursor)?);
        cache.dim_bufman.close_cursor(cursor)?;

        Ok(Self {
            root,
            cache,
            total_documents_count,
            data_file_parts,
        })
    }
}
