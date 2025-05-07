use super::buffered_io::{BufIoError, BufferManager, BufferManagerFactory};
use super::common::TSHashTable;
use super::file_persist::{read_prop_metadata_from_file, read_prop_value_from_file};
use super::inverted_index::InvertedIndexNodeData;
use super::lru_cache::LRUCache;
use super::prob_lazy_load::lazy_item::{FileIndex, ProbLazyItem};
use super::prob_node::{ProbNode, SharedNode};
use super::serializer::hnsw::HNSWIndexSerialize;
use super::serializer::inverted::InvertedIndexSerialize;
use super::serializer::tf_idf::TFIDFIndexSerialize;
use super::tf_idf_index::TFIDFIndexNodeData;
use super::types::*;
use crate::indexes::hnsw::offset_counter::IndexFileId;
use dashmap::DashMap;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex, RwLock, Weak};

pub struct HNSWIndexCache {
    pub registry: LRUCache<u64, SharedNode>,
    props_registry: DashMap<u64, Weak<NodePropValue>>,
    pub bufmans: Arc<BufferManagerFactory<IndexFileId>>,
    pub prop_file: RwLock<File>,
    loading_items: TSHashTable<u64, Arc<Mutex<bool>>>,
    pub distance_metric: Arc<RwLock<DistanceMetric>>,
}

unsafe impl Send for HNSWIndexCache {}
unsafe impl Sync for HNSWIndexCache {}

impl HNSWIndexCache {
    pub fn new(
        bufmans: Arc<BufferManagerFactory<IndexFileId>>,
        prop_file: RwLock<File>,
        distance_metric: Arc<RwLock<DistanceMetric>>,
    ) -> Self {
        let registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);
        let props_registry = DashMap::new();

        Self {
            registry,
            props_registry,
            bufmans,
            prop_file,
            distance_metric,
            loading_items: TSHashTable::new(16),
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn unload(&self, item: SharedNode) -> Result<(), BufIoError> {
        if item.is_null() {
            return Ok(());
        }
        let item_ref = unsafe { &*item };
        let combined_index = Self::combine_index(&item_ref.file_index);
        self.registry.remove(&combined_index);
        unsafe {
            drop(Box::from_raw(item));
        }
        Ok(())
    }

    pub fn flush_all(&self) -> Result<(), BufIoError> {
        self.bufmans.flush_all()?;
        self.prop_file
            .write()
            .map_err(|_| BufIoError::Locking)?
            .flush()
            .map_err(BufIoError::Io)
    }

    pub fn get_prop(
        &self,
        offset: FileOffset,
        length: BytesToRead,
    ) -> Result<Arc<NodePropValue>, BufIoError> {
        let key = Self::get_prop_key(offset, length);
        if let Some(prop) = self
            .props_registry
            .get(&key)
            .and_then(|prop| prop.upgrade())
        {
            return Ok(prop);
        }
        let mut prop_file_guard = self.prop_file.write().unwrap();
        let prop = Arc::new(read_prop_value_from_file(
            (offset, length),
            &mut prop_file_guard,
        )?);
        drop(prop_file_guard);
        let weak = Arc::downgrade(&prop);
        self.props_registry.insert(key, weak);
        Ok(prop)
    }

    /// Reads prop_metadata from the prop file
    ///
    /// @NOTE: Right now, every call reads from the prop file, there's
    /// no caching implemented
    ///
    /// @TODO: Implement caching for prop_metadata as well
    pub fn get_prop_metadata(
        &self,
        offset: FileOffset,
        length: BytesToRead,
    ) -> Result<Arc<NodePropMetadata>, BufIoError> {
        let mut prop_file_guard = self.prop_file.write().unwrap();
        let metadata = Arc::new(read_prop_metadata_from_file(
            (offset, length),
            &mut prop_file_guard,
        )?);
        drop(prop_file_guard);
        Ok(metadata)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn insert_lazy_object(&self, item: SharedNode) {
        let item_ref = unsafe { &*item };
        let combined_index = Self::combine_index(&item_ref.file_index);
        if let Some(node) = item_ref.get_lazy_data() {
            let prop_key =
                Self::get_prop_key(node.prop_value.location.0, node.prop_value.location.1);
            self.props_registry
                .insert(prop_key, Arc::downgrade(&node.prop_value));
        }
        self.registry.insert(combined_index, item);
    }

    pub fn get_object(&self, file_index: FileIndex) -> Result<SharedNode, BufIoError> {
        let combined_index = Self::combine_index(&file_index);

        if let Some(item) = self.registry.get(&combined_index) {
            return Ok(item);
        }

        let mut mutex = self
            .loading_items
            .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
        let mut load_complete = mutex.lock().unwrap();

        loop {
            // check again
            if let Some(item) = self.registry.get(&combined_index) {
                return Ok(item);
            }

            // another thread loaded the data but its not in the registry (got evicted), retry
            if *load_complete {
                drop(load_complete);
                mutex = self
                    .loading_items
                    .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
                load_complete = mutex.lock().unwrap();
                continue;
            }

            break;
        }

        let bufman = self.bufmans.get(file_index.file_id)?;
        let mut pending_items = FxHashMap::default();

        let data = ProbNode::deserialize(
            &bufman,
            file_index.offset,
            file_index.file_id,
            self,
            &mut pending_items,
        )?;

        let FileIndex { offset, file_id } = file_index;
        let item = ProbLazyItem::new(data, file_id, offset);

        self.registry.insert(combined_index, item);

        *load_complete = true;
        self.loading_items.delete(&combined_index);

        Ok(item)
    }

    pub fn combine_index(file_index: &FileIndex) -> u64 {
        ((file_index.offset.0 as u64) << 32) | (*file_index.file_id as u64)
    }

    pub fn get_prop_key(
        FileOffset(file_offset): FileOffset,
        BytesToRead(length): BytesToRead,
    ) -> u64 {
        ((file_offset as u64) << 32) | (length as u64)
    }
}

pub struct InvertedIndexCache {
    registry: LRUCache<u64, *mut ProbLazyItem<InvertedIndexNodeData>>,
    pub dim_bufman: Arc<BufferManager>,
    pub data_bufmans: Arc<BufferManagerFactory<u8>>,
    loading_data: TSHashTable<u64, Arc<Mutex<bool>>>,
    pub data_file_parts: u8,
}

unsafe impl Send for InvertedIndexCache {}
unsafe impl Sync for InvertedIndexCache {}

impl InvertedIndexCache {
    pub fn new(
        dim_bufman: Arc<BufferManager>,
        data_bufmans: Arc<BufferManagerFactory<u8>>,
        data_file_parts: u8,
    ) -> Self {
        let data_registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);

        Self {
            registry: data_registry,
            dim_bufman,
            data_bufmans,
            loading_data: TSHashTable::new(16),
            data_file_parts,
        }
    }

    pub fn get_data(
        &self,
        file_offset: FileOffset,
        data_file_idx: u8,
    ) -> Result<*mut ProbLazyItem<InvertedIndexNodeData>, BufIoError> {
        let combined_index = Self::combine_index(file_offset, 0);

        if let Some(item) = self.registry.get(&combined_index) {
            return Ok(item);
        }

        let mut mutex = self
            .loading_data
            .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
        let mut load_complete = mutex.lock().unwrap();

        loop {
            // check again
            if let Some(item) = self.registry.get(&combined_index) {
                return Ok(item);
            }

            // another thread loaded the data but its not in the registry (got evicted), retry
            if *load_complete {
                drop(load_complete);
                mutex = self
                    .loading_data
                    .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
                load_complete = mutex.lock().unwrap();
                continue;
            }

            break;
        }

        let data = InvertedIndexNodeData::deserialize(
            &self.dim_bufman,
            &self.data_bufmans,
            file_offset,
            data_file_idx,
            self.data_file_parts,
            self,
        )?;

        let item = ProbLazyItem::new(data, IndexFileId::invalid(), file_offset);

        self.registry.insert(combined_index, item);

        *load_complete = true;
        self.loading_data.delete(&combined_index);

        Ok(item)
    }

    pub fn combine_index(file_offset: FileOffset, data_file_idx: u8) -> u64 {
        ((data_file_idx as u64) << 32) | file_offset.0 as u64
    }

    #[allow(unused)]
    pub fn load_item<T: InvertedIndexSerialize>(
        &self,
        file_offset: FileOffset,
        data_file_idx: u8,
    ) -> Result<T, BufIoError> {
        T::deserialize(
            &self.dim_bufman,
            &self.data_bufmans,
            file_offset,
            data_file_idx,
            self.data_file_parts,
            self,
        )
    }

    pub fn flush_all(&self) -> Result<(), BufIoError> {
        self.dim_bufman.flush()?;
        self.data_bufmans.flush_all()
    }
}

pub struct TFIDFIndexCache {
    registry: LRUCache<u64, *mut ProbLazyItem<TFIDFIndexNodeData>>,
    pub dim_bufman: Arc<BufferManager>,
    pub data_bufmans: Arc<BufferManagerFactory<u8>>,
    pub offset_counter: AtomicU32,
    loading_data: TSHashTable<u64, Arc<Mutex<bool>>>,
    pub data_file_parts: u8,
}

unsafe impl Send for TFIDFIndexCache {}
unsafe impl Sync for TFIDFIndexCache {}

impl TFIDFIndexCache {
    pub fn new(
        dim_bufman: Arc<BufferManager>,
        data_bufmans: Arc<BufferManagerFactory<u8>>,
        offset_counter: AtomicU32,
        data_file_parts: u8,
    ) -> Self {
        let data_registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);

        Self {
            registry: data_registry,
            dim_bufman,
            data_bufmans,
            offset_counter,
            loading_data: TSHashTable::new(16),
            data_file_parts,
        }
    }

    pub fn get_data(
        &self,
        file_offset: FileOffset,
        data_file_idx: u8,
    ) -> Result<*mut ProbLazyItem<TFIDFIndexNodeData>, BufIoError> {
        let combined_index = Self::combine_index(file_offset, 0);

        if let Some(item) = self.registry.get(&combined_index) {
            return Ok(item);
        }

        let mut mutex = self
            .loading_data
            .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
        let mut load_complete = mutex.lock().unwrap();

        loop {
            // check again
            if let Some(item) = self.registry.get(&combined_index) {
                return Ok(item);
            }

            // another thread loaded the data but its not in the registry (got evicted), retry
            if *load_complete {
                drop(load_complete);
                mutex = self
                    .loading_data
                    .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
                load_complete = mutex.lock().unwrap();
                continue;
            }

            break;
        }

        let data = TFIDFIndexNodeData::deserialize(
            &self.dim_bufman,
            &self.data_bufmans,
            file_offset,
            data_file_idx,
            self.data_file_parts,
            self,
        )?;

        let item = ProbLazyItem::new(data, IndexFileId::invalid(), file_offset);

        self.registry.insert(combined_index, item);

        *load_complete = true;
        self.loading_data.delete(&combined_index);

        Ok(item)
    }

    pub fn combine_index(file_offset: FileOffset, data_file_idx: u8) -> u64 {
        ((data_file_idx as u64) << 32) | file_offset.0 as u64
    }

    #[allow(unused)]
    pub fn load_item<T: TFIDFIndexSerialize>(
        &self,
        file_offset: FileOffset,
        data_file_idx: u8,
    ) -> Result<T, BufIoError> {
        T::deserialize(
            &self.dim_bufman,
            &self.data_bufmans,
            file_offset,
            data_file_idx,
            self.data_file_parts,
            self,
        )
    }

    pub fn flush_all(&self) -> Result<(), BufIoError> {
        self.dim_bufman.flush()?;
        self.data_bufmans.flush_all()
    }
}
