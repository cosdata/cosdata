use crate::indexes::hnsw::offset_counter::IndexFileId;

use super::buffered_io::{BufIoError, BufferManager, BufferManagerFactory, FilelessBufferManager};
use super::common::TSHashTable;
use super::file_persist::{read_prop_metadata_from_file, read_prop_value_from_file};
use super::inverted_index::InvertedIndexNodeData;
use super::lazy_item::{FileIndex, LazyItem};
use super::lru_cache::LRUCache;
use super::prob_node::{ProbNode, SharedNode};
use super::serializer::hnsw::HNSWIndexSerialize;
use super::serializer::inverted::InvertedIndexSerialize;
use super::serializer::tf_idf::TFIDFIndexSerialize;
use super::tf_idf_index::TFIDFIndexNodeData;
use super::types::*;
use super::versioning::VersionNumber;
use dashmap::DashMap;
use rustc_hash::FxHashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::TryLockError;
use std::sync::{Arc, Mutex, RwLock, Weak};

pub struct HNSWIndexCache {
    metadata_registry: DashMap<u64, Weak<NodePropMetadata>>,
    pub registry: LRUCache<u64, SharedNode>,
    props_registry: DashMap<u64, Weak<NodePropValue>>,
    pub bufmans: BufferManagerFactory<IndexFileId>,
    pub latest_version_links_bufman: FilelessBufferManager,
    root_path: PathBuf,
    enable_context_history: bool,
    pub prop_file: RwLock<File>,
    loading_items: TSHashTable<u64, Arc<Mutex<bool>>>,
    pub distance_metric: Arc<RwLock<DistanceMetric>>,
    // A global lock to prevent deadlocks during batch loading of cache entries when `max_loads > 1`.
    //
    // This lock ensures that only one thread is allowed to load large batches of nodes (where `max_loads > 1`)
    // at any given time. If multiple threads attempt to load interconnected nodes in parallel with high `max_loads`,
    // it can lead to a deadlock situation due to circular dependencies between the locks. By serializing access to
    // large batch loads, this mutex ensures that only one thread can initiate a batch load with a high `max_loads`
    // value, preventing such circular waiting conditions. Threads with `max_loads = 1` can still load nodes in parallel
    // without causing conflicts, allowing for efficient loading of smaller batches.
    batch_load_lock: Mutex<()>,
}

unsafe impl Send for HNSWIndexCache {}
unsafe impl Sync for HNSWIndexCache {}

impl HNSWIndexCache {
    pub fn new(
        bufmans: BufferManagerFactory<IndexFileId>,
        latest_version_links_bufman: FilelessBufferManager,
        root_path: PathBuf,
        enable_context_history: bool,
        prop_file: RwLock<File>,
        distance_metric: Arc<RwLock<DistanceMetric>>,
    ) -> Self {
        let registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);
        let props_registry = DashMap::new();
        let metadata_registry = DashMap::new();
        Self {
            registry,
            props_registry,
            bufmans,
            latest_version_links_bufman,
            root_path,
            enable_context_history,
            prop_file,
            distance_metric,
            loading_items: TSHashTable::new(16),
            batch_load_lock: Mutex::new(()),
            metadata_registry,
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

    pub fn flush_all(&self, version: VersionNumber) -> Result<(), BufIoError> {
        self.bufmans.flush_all()?;
        self.prop_file
            .write()
            .map_err(|_| BufIoError::Locking)?
            .flush()
            .map_err(BufIoError::Io)?;
        if self.enable_context_history {
            self.latest_version_links_bufman
                .flush_versioned(|region_id| {
                    self.root_path
                        .join(format!("{}-{}.ptr", region_id, *version))
                })
        } else {
            let mut file = OpenOptions::new()
                .read(false)
                .write(true)
                .create(true)
                .truncate(false)
                .open(self.root_path.join("nodes.ptr"))?;
            self.latest_version_links_bufman.flush(&mut file)
        }
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
    pub fn get_prop_metadata(
        &self,
        offset: FileOffset,
        length: BytesToRead,
    ) -> Result<Arc<NodePropMetadata>, BufIoError> {
        let key = Self::get_prop_key(offset, length);
        if let Some(metadata) = self
            .metadata_registry
            .get(&key)
            .and_then(|metadata| metadata.upgrade())
        {
            return Ok(metadata);
        }

        let mut prop_file_guard = self.prop_file.write().unwrap();
        let metadata = Arc::new(read_prop_metadata_from_file(
            (offset, length),
            &mut prop_file_guard,
        )?);
        drop(prop_file_guard);
        let weak = Arc::downgrade(&metadata);
        self.metadata_registry.insert(key, weak);
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
            if let Some(ref metadata) = node.prop_metadata {
                let metadata_key = Self::get_prop_key(metadata.location.0, metadata.location.1);
                self.metadata_registry
                    .insert(metadata_key, Arc::downgrade(metadata));
            }
        }
        self.registry.insert(combined_index, item);
    }

    pub fn get_lazy_object(
        &self,
        file_index: FileIndex<IndexFileId>,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<SharedNode, BufIoError> {
        let combined_index = Self::combine_index(&file_index);

        if let Some(item) = self.registry.get(&combined_index) {
            return Ok(item);
        }

        if max_loads == 0 || !skipm.insert(combined_index) {
            return Ok(LazyItem::new_pending(file_index));
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
        let data = ProbNode::deserialize(
            &bufman,
            &self.latest_version_links_bufman,
            file_index,
            self,
            max_loads - 1,
            skipm,
        )?;
        let item = LazyItem::new(data, file_index.file_id, file_index.offset);

        self.registry.insert(combined_index, item);
        *load_complete = true;
        self.loading_items.delete(&combined_index);

        Ok(item)
    }

    // Retrieves an object from the cache, attempting to batch load if possible, based on the state of the batch load lock.
    //
    // This function first attempts to acquire the `batch_load_lock` using a non-blocking `try_lock`. If successful,
    // it sets a high `max_loads` value (1000), allowing for a larger batch load. This is the preferred scenario where
    // the system is capable of performing a more efficient batch load, loading multiple nodes at once. If the lock is
    // already held (i.e., another thread is performing a large batch load), the function falls back to a lower `max_loads`
    // value (1), effectively loading nodes one at a time to avoid blocking or deadlocking.
    //
    // The key idea here is to **always attempt to load as many nodes as possible** (with `max_loads = 1000`) unless
    // another thread is already performing a large load, in which case the function resorts to a smaller load size.
    // This dynamic loading strategy balances efficient batch loading with the need to avoid blocking or deadlocks in high-concurrency situations.
    //
    // After determining the appropriate `max_loads`, the function proceeds by calling `get_lazy_object`, which handles
    // the actual loading process, and retrieves the lazy-loaded data.
    pub fn get_object(&self, file_index: FileIndex<IndexFileId>) -> Result<SharedNode, BufIoError> {
        let (_lock, max_loads) = match self.batch_load_lock.try_lock() {
            Ok(lock) => (Some(lock), 1000),
            Err(TryLockError::Poisoned(poison_err)) => panic!("lock error: {}", poison_err),
            Err(TryLockError::WouldBlock) => (None, 1),
        };
        self.get_lazy_object(file_index, max_loads, &mut FxHashSet::default())
    }

    pub fn combine_index(file_index: &FileIndex<IndexFileId>) -> u64 {
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
    registry: LRUCache<u64, *mut LazyItem<InvertedIndexNodeData, ()>>,
    pub dim_bufman: Arc<BufferManager>,
    pub data_bufmans: Arc<BufferManagerFactory<VersionNumber>>,
    loading_data: TSHashTable<u64, Arc<Mutex<bool>>>,
}

unsafe impl Send for InvertedIndexCache {}
unsafe impl Sync for InvertedIndexCache {}

impl InvertedIndexCache {
    pub fn new(
        dim_bufman: Arc<BufferManager>,
        data_bufmans: Arc<BufferManagerFactory<VersionNumber>>,
    ) -> Self {
        let data_registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);

        Self {
            registry: data_registry,
            dim_bufman,
            data_bufmans,
            loading_data: TSHashTable::new(16),
        }
    }

    pub fn get_data(
        &self,
        file_offset: FileOffset,
    ) -> Result<*mut LazyItem<InvertedIndexNodeData, ()>, BufIoError> {
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
            VersionNumber::from(u32::MAX), // not used
            self,
        )?;

        let item = LazyItem::new(data, (), file_offset);

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
        version: VersionNumber,
    ) -> Result<T, BufIoError> {
        T::deserialize(
            &self.dim_bufman,
            &self.data_bufmans,
            file_offset,
            version,
            self,
        )
    }

    pub fn flush_all(&self) -> Result<(), BufIoError> {
        self.dim_bufman.flush()?;
        self.data_bufmans.flush_all()
    }
}

pub struct TFIDFIndexCache {
    registry: LRUCache<u64, *mut LazyItem<TFIDFIndexNodeData, ()>>,
    pub dim_bufman: Arc<BufferManager>,
    pub data_bufmans: Arc<BufferManagerFactory<VersionNumber>>,
    pub offset_counter: AtomicU32,
    loading_data: TSHashTable<u64, Arc<Mutex<bool>>>,
}

unsafe impl Send for TFIDFIndexCache {}
unsafe impl Sync for TFIDFIndexCache {}

impl TFIDFIndexCache {
    pub fn new(
        dim_bufman: Arc<BufferManager>,
        data_bufmans: Arc<BufferManagerFactory<VersionNumber>>,
        offset_counter: AtomicU32,
    ) -> Self {
        let data_registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);

        Self {
            registry: data_registry,
            dim_bufman,
            data_bufmans,
            offset_counter,
            loading_data: TSHashTable::new(16),
        }
    }

    pub fn get_data(
        &self,
        file_offset: FileOffset,
    ) -> Result<*mut LazyItem<TFIDFIndexNodeData, ()>, BufIoError> {
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
            VersionNumber::from(u32::MAX), // not used
            self,
        )?;

        let item = LazyItem::new(data, (), file_offset);

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
        version: VersionNumber,
    ) -> Result<T, BufIoError> {
        T::deserialize(
            &self.dim_bufman,
            &self.data_bufmans,
            file_offset,
            version,
            self,
        )
    }

    pub fn flush_all(&self) -> Result<(), BufIoError> {
        self.dim_bufman.flush()?;
        self.data_bufmans.flush_all()
    }
}
