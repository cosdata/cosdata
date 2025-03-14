use super::buffered_io::{BufIoError, BufferManager, BufferManagerFactory};
use super::common::TSHashTable;
use super::file_persist::read_prop_from_file;
use super::fixedset::VersionedInvertedFixedSetIndex;
use super::inverted_index::InvertedIndexNodeData;
use super::lru_cache::LRUCache;
use super::prob_lazy_load::lazy_item::{FileIndex, ProbLazyItem, ProbLazyItemState, ReadyState};
use super::prob_node::{ProbNode, SharedNode};
use super::serializer::hnsw::HNSWIndexSerialize;
use super::serializer::inverted::InvertedIndexSerialize;
use super::types::*;
use super::versioning::Hash;
use dashmap::DashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::sync::TryLockError;
use std::sync::{Arc, Mutex, RwLock, Weak};

pub struct HNSWIndexCache {
    registry: LRUCache<u64, SharedNode>,
    props_registry: DashMap<u64, Weak<NodeProp>>,
    pub bufmans: Arc<BufferManagerFactory<Hash>>,
    pub level_0_bufmans: Arc<BufferManagerFactory<Hash>>,
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
        bufmans: Arc<BufferManagerFactory<Hash>>,
        level_0_bufmans: Arc<BufferManagerFactory<Hash>>,
        prop_file: RwLock<File>,
        distance_metric: Arc<RwLock<DistanceMetric>>,
    ) -> Self {
        let registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);
        let props_registry = DashMap::new();

        Self {
            registry,
            props_registry,
            bufmans,
            level_0_bufmans,
            prop_file,
            distance_metric,
            loading_items: TSHashTable::new(16),
            batch_load_lock: Mutex::new(()),
        }
    }

    pub fn flush_all(&self) -> Result<(), BufIoError> {
        self.bufmans.flush_all()?;
        self.level_0_bufmans.flush_all()?;
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
    ) -> Result<Arc<NodeProp>, BufIoError> {
        let key = Self::get_prop_key(offset, length);
        if let Some(prop) = self
            .props_registry
            .get(&key)
            .and_then(|prop| prop.upgrade())
        {
            return Ok(prop);
        }
        let mut prop_file_guard = self.prop_file.write().unwrap();
        let prop = Arc::new(read_prop_from_file((offset, length), &mut prop_file_guard)?);
        drop(prop_file_guard);
        let weak = Arc::downgrade(&prop);
        self.props_registry.insert(key, weak);
        Ok(prop)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn insert_lazy_object(&self, version: Hash, item: SharedNode) {
        let item_ref = unsafe { &*item };
        let combined_index =
            ((item_ref.get_file_index().offset.0 as u64) << 32) | (*version as u64);
        if let Some(node) = item_ref.get_lazy_data() {
            let prop_key = Self::get_prop_key(node.prop.location.0, node.prop.location.1);
            self.props_registry
                .insert(prop_key, Arc::downgrade(&node.prop));
        }
        self.registry.insert(combined_index, item);
    }

    pub fn force_load_single_object(
        &self,
        file_index: FileIndex,
        is_level_0: bool,
    ) -> Result<SharedNode, BufIoError> {
        let combined_index = Self::combine_index(&file_index, is_level_0);
        let mut skipm = HashSet::new();
        skipm.insert(combined_index);
        let bufmans = if is_level_0 {
            &self.level_0_bufmans
        } else {
            &self.bufmans
        };
        let data = ProbNode::deserialize(bufmans, file_index, self, 0, &mut skipm, is_level_0)?;
        let FileIndex {
            offset: file_offset,
            version_number,
            version_id,
        } = file_index;
        let state = ProbLazyItemState::Ready(ReadyState {
            data,
            file_offset,
            version_id,
            version_number,
        });

        let item = ProbLazyItem::new_from_state(state, is_level_0);

        self.registry.insert(combined_index, item);

        Ok(item)
    }

    pub fn get_lazy_object(
        &self,
        file_index: FileIndex,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<SharedNode, BufIoError> {
        let combined_index = Self::combine_index(&file_index, is_level_0);

        if let Some(item) = self.registry.get(&combined_index) {
            return Ok(item);
        }

        if max_loads == 0 || !skipm.insert(combined_index) {
            return Ok(ProbLazyItem::new_pending(file_index, is_level_0));
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

        let FileIndex {
            offset: file_offset,
            version_number,
            version_id,
        } = file_index;

        let bufmans = if is_level_0 {
            &self.level_0_bufmans
        } else {
            &self.bufmans
        };

        let data =
            ProbNode::deserialize(bufmans, file_index, self, max_loads - 1, skipm, is_level_0)?;
        let state = ProbLazyItemState::Ready(ReadyState {
            data,
            file_offset,
            version_id,
            version_number,
        });

        let item = ProbLazyItem::new_from_state(state, is_level_0);

        self.registry.insert(combined_index, item);

        *load_complete = true;
        self.loading_items.delete(&combined_index);

        Ok(item)
    }

    pub fn load_region(
        &self,
        region_start: u32,
        version_number: u16,
        version_id: Hash,
        node_size: u32,
        is_level_0: bool,
    ) -> Result<Vec<SharedNode>, BufIoError> {
        let bufman = if is_level_0 {
            self.level_0_bufmans.get(version_id)?
        } else {
            self.bufmans.get(version_id)?
        };
        let file_size = bufman.file_size();
        if region_start as u64 > file_size {
            return Ok(Vec::new());
        }
        println!(
            "Loading region: {}, version: {}, is_level_0: {}",
            region_start, version_number, is_level_0
        );
        let cap = ((file_size - region_start as u64) / node_size as u64).min(1000) as usize;
        let mut nodes = Vec::with_capacity(cap);
        for i in 0..1000 {
            let offset = FileOffset(i * node_size + region_start);
            if offset.0 as u64 >= file_size {
                break;
            }
            let file_index = FileIndex {
                offset,
                version_number,
                version_id,
            };
            let node = self.force_load_single_object(file_index, is_level_0)?;
            nodes.push(node);
        }
        Ok(nodes)
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
    pub fn get_object(
        &self,
        file_index: FileIndex,
        is_level_0: bool,
    ) -> Result<SharedNode, BufIoError> {
        let (_lock, max_loads) = match self.batch_load_lock.try_lock() {
            Ok(lock) => (Some(lock), 1000),
            Err(TryLockError::Poisoned(poison_err)) => panic!("lock error: {}", poison_err),
            Err(TryLockError::WouldBlock) => (None, 1),
        };
        self.get_lazy_object(file_index, max_loads, &mut HashSet::new(), is_level_0)
    }

    pub fn combine_index(file_index: &FileIndex, is_level_0: bool) -> u64 {
        let level_bit = if is_level_0 { 1u64 << 63 } else { 0 };
        ((file_index.offset.0 as u64) << 32) | (*file_index.version_id as u64) | level_bit
    }

    pub fn get_prop_key(
        FileOffset(file_offset): FileOffset,
        BytesToRead(length): BytesToRead,
    ) -> u64 {
        ((file_offset as u64) << 32) | (length as u64)
    }

    #[allow(unused)]
    pub fn load_item<T: HNSWIndexSerialize>(
        &self,
        file_index: FileIndex,
        is_level_0: bool,
    ) -> Result<T, BufIoError> {
        let mut skipm: HashSet<u64> = HashSet::new();

        let bufmans = if is_level_0 {
            &self.level_0_bufmans
        } else {
            &self.bufmans
        };

        T::deserialize(bufmans, file_index, self, 1000, &mut skipm, is_level_0)
    }
}

pub struct InvertedIndexCache {
    data_registry: LRUCache<u64, *mut ProbLazyItem<InvertedIndexNodeData>>,
    sets_registry: LRUCache<u64, *mut ProbLazyItem<VersionedInvertedFixedSetIndex>>,
    pub dim_bufman: Arc<BufferManager>,
    pub data_bufmans: Arc<BufferManagerFactory<u8>>,
    loading_data: TSHashTable<u64, Arc<Mutex<bool>>>,
    loading_sets: TSHashTable<u64, Arc<Mutex<bool>>>,
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
        let sets_registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);

        Self {
            data_registry,
            sets_registry,
            dim_bufman,
            data_bufmans,
            loading_data: TSHashTable::new(16),
            loading_sets: TSHashTable::new(16),
            data_file_parts,
        }
    }

    pub fn get_data(
        &self,
        file_offset: FileOffset,
        data_file_idx: u8,
    ) -> Result<*mut ProbLazyItem<InvertedIndexNodeData>, BufIoError> {
        let combined_index = Self::combine_index(file_offset, 0);

        if let Some(item) = self.data_registry.get(&combined_index) {
            return Ok(item);
        }

        let mut mutex = self
            .loading_data
            .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
        let mut load_complete = mutex.lock().unwrap();

        loop {
            // check again
            if let Some(item) = self.data_registry.get(&combined_index) {
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
        let state = ProbLazyItemState::Ready(ReadyState {
            data,
            file_offset,
            version_id: 0.into(),
            version_number: 0,
        });

        let item = ProbLazyItem::new_from_state(state, false);

        self.data_registry.insert(combined_index, item);

        *load_complete = true;
        self.loading_data.delete(&combined_index);

        Ok(item)
    }

    pub fn get_sets(
        &self,
        file_offset: FileOffset,
        data_file_idx: u8,
    ) -> Result<*mut ProbLazyItem<VersionedInvertedFixedSetIndex>, BufIoError> {
        let combined_index = Self::combine_index(file_offset, 0);

        if let Some(item) = self.sets_registry.get(&combined_index) {
            return Ok(item);
        }

        let mut mutex = self
            .loading_data
            .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
        let mut load_complete = mutex.lock().unwrap();

        loop {
            // check again
            if let Some(item) = self.sets_registry.get(&combined_index) {
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

        let dim_cursor = self.dim_bufman.open_cursor()?;
        self.dim_bufman
            .seek_with_cursor(dim_cursor, file_offset.0 as u64)?;
        let data_offset = self.dim_bufman.read_u32_with_cursor(dim_cursor)?;
        self.dim_bufman.close_cursor(dim_cursor)?;

        let data = VersionedInvertedFixedSetIndex::deserialize(
            &self.dim_bufman,
            &self.data_bufmans,
            FileOffset(data_offset),
            data_file_idx,
            self.data_file_parts,
            self,
        )?;
        let state = ProbLazyItemState::Ready(ReadyState {
            data,
            file_offset,
            version_id: 0.into(),
            version_number: 0,
        });

        let item = ProbLazyItem::new_from_state(state, false);

        self.sets_registry.insert(combined_index, item);

        *load_complete = true;
        self.loading_sets.delete(&combined_index);

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
}
