use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::file_persist::*;
use super::lazy_load::{FileIndex, LazyItem, LazyItemVec, VectorData};
use super::lru_cache::LRUCache;
use super::prob_lazy_load::lazy_item::{ProbLazyItem, ProbLazyItemState};
use super::prob_lazy_load::lazy_item_array::ProbLazyItemArray;
use super::serializer::prob::{lazy_item_deserialize_impl, ProbSerialize, UpdateSerialized};
use super::serializer::CustomSerialize;
use super::types::*;
use crate::models::lru_cache::CachedValue;
use crate::storage::inverted_index_old::InvertedIndexItem;
use crate::storage::inverted_index_sparse_ann::{
    InvertedIndexSparseAnn, InvertedIndexSparseAnnNode,
};
use crate::storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasic;
use crate::storage::inverted_index_sparse_ann_basic::{
    InvertedIndexSparseAnnNodeBasicDashMap, InvertedIndexSparseAnnNodeBasicTSHashmap,
};
use crate::storage::inverted_index_sparse_ann_new_ds::InvertedIndexNewDSNode;
use crate::storage::Storage;
use arcshift::ArcShift;
use dashmap::DashSet;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::cell::Cell;
use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::ptr;
use std::sync::atomic::Ordering;
use std::sync::{atomic::AtomicBool, Arc, RwLock};

macro_rules! define_cache_items {
    ($($variant:ident = $type:ty),+ $(,)?) => {
        #[derive(Clone)]
        pub enum CacheItem {
            $($variant(LazyItem<$type>)),+
        }


        pub trait Cacheable: Clone + 'static {
            fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>>;
            fn into_cache_item(item: LazyItem<Self>) -> CacheItem;
        }

        $(
            impl Cacheable for $type {
                fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
                    if let CacheItem::$variant(item) = cache_item {
                        Some(item)
                    } else {
                        None
                    }
                }

                fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
                    CacheItem::$variant(item)
                }
            }
        )+
    };
}

define_cache_items! {
    MergedNode = MergedNode,
    Storage = Storage,
    InvertedIndexItemWithStorage = InvertedIndexItem<Storage>,
    Float = f32,
    Unsigned32 = u32,
    InvertedIndexItemWithFloat = InvertedIndexItem<f32>,
    InvertedIndexSparseAnnNode = InvertedIndexSparseAnnNode,
    InvertedIndexSparseAnnNodeBasic = InvertedIndexSparseAnnNodeBasic,
    InvertedIndexSparseAnn = InvertedIndexSparseAnn,
    InvertedIndexSparseAnnNodeBasicTSHashmap = InvertedIndexSparseAnnNodeBasicTSHashmap,
    InvertedIndexSparseAnnNodeBasicDashMap = InvertedIndexSparseAnnNodeBasicDashMap,
    InvertedIndexNewDSNode = InvertedIndexNewDSNode,
    VectorData = STM<VectorData>,
    ProbNode = ProbNode,
}

pub struct NodeRegistry {
    cuckoo_filter: RwLock<CuckooFilter<u64>>,
    registry: LRUCache<u64, CacheItem>,
    bufmans: Arc<BufferManagerFactory>,
}

impl NodeRegistry {
    pub fn new(cuckoo_filter_capacity: usize, bufmans: Arc<BufferManagerFactory>) -> Self {
        let cuckoo_filter = CuckooFilter::new(cuckoo_filter_capacity);
        let registry = LRUCache::with_prob_eviction(1000, 0.03125);
        NodeRegistry {
            cuckoo_filter: RwLock::new(cuckoo_filter),
            registry,
            bufmans,
        }
    }

    pub fn get_bufmans(&self) -> Arc<BufferManagerFactory> {
        self.bufmans.clone()
    }

    pub fn get_object<T: Cacheable, F>(
        self: Arc<Self>,
        file_index: FileIndex,
        load_function: F,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<LazyItem<T>, BufIoError>
    where
        F: Fn(
            Arc<BufferManagerFactory>,
            FileIndex,
            Arc<Self>,
            u16,
            &mut HashSet<u64>,
        ) -> Result<LazyItem<T>, BufIoError>,
    {
        println!(
            "get_object called with file_index: {:?}, max_loads: {}",
            file_index, max_loads
        );

        let combined_index = Self::combine_index(&file_index);

        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            println!("Acquired read lock on cuckoo_filter");

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&combined_index) {
                println!("FileIndex found in cuckoo_filter");
                if let Some(obj) = self.registry.get(&combined_index) {
                    if let Some(item) = T::from_cache_item(obj) {
                        println!("Object found in registry, returning");
                        return Ok(item);
                    }
                } else {
                    println!("Object not found in registry despite being in cuckoo_filter");
                }
            } else {
                println!("FileIndex not found in cuckoo_filter");
            }
        }
        println!("Released read lock on cuckoo_filter");

        let (version_id, version_number) = if let FileIndex::Valid {
            version_id,
            version_number,
            ..
        } = &file_index
        {
            (*version_id, *version_number)
        } else {
            (0.into(), 0)
        };

        if max_loads == 0 || !skipm.insert(combined_index) {
            println!("Either max_loads hit 0 or loop detected, returning LazyItem with no data");
            return Ok(LazyItem::Valid {
                data: ArcShift::new(None),
                file_index: ArcShift::new(Some(file_index)),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions: LazyItemVec::new(),
                version_id,
                version_number,
                serialized_flag: Arc::new(AtomicBool::new(true)),
            });
        }

        println!("Calling load_function");
        let item = load_function(
            self.bufmans.clone(),
            file_index.clone(),
            self.clone(),
            max_loads - 1,
            skipm,
        )?;
        println!("load_function returned successfully");

        println!("Trying to get or insert item into registry");
        let cached_item = self
            .registry
            .get_or_insert::<BufIoError>(combined_index.clone(), || Ok(T::into_cache_item(item)))?;

        match cached_item {
            CachedValue::Hit(item) => {
                println!("Object found in registry after load, returning");
                Ok(T::from_cache_item(item).unwrap())
            }
            CachedValue::Miss(item) => {
                println!("Inserting key into cuckoo_filter");
                self.cuckoo_filter.write().unwrap().insert(&combined_index);

                println!("Returning newly created LazyItem");
                Ok(T::from_cache_item(item).unwrap())
            }
        }
    }

    pub fn load_item<T: CustomSerialize>(
        self: Arc<Self>,
        file_index: FileIndex,
    ) -> Result<T, BufIoError> {
        let mut skipm: HashSet<u64> = HashSet::new();

        if file_index == FileIndex::Invalid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )
            .into());
        };

        T::deserialize(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            1000,
            &mut skipm,
        )
    }

    pub fn combine_index(file_index: &FileIndex) -> u64 {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => ((offset.0 as u64) << 32) | (**version_id as u64),
            FileIndex::Invalid => u64::MAX, // Use max u64 value for Invalid
        }
    }

    // pub fn split_combined_index(combined: u64) -> FileIndex {
    //     if combined == u64::MAX {
    //         FileIndex::Invalid
    //     } else {
    //         FileIndex::Valid {
    //             offset: FileOffset((combined >> 32) as u32),
    //             version: (combined as u32).into(),
    //         }
    //     }
    // }
}

pub fn load_cache() {
    // TODO: include db name in the path
    let bufmans = Arc::new(BufferManagerFactory::new(
        Path::new(".").into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));

    // TODO: fill appropriate version info
    let file_index = FileIndex::Valid {
        offset: FileOffset(0),
        version_id: 0.into(),
        version_number: 0,
    };
    let cache = Arc::new(NodeRegistry::new(1000, bufmans));
    match read_node_from_file(file_index.clone(), cache) {
        Ok(_) => println!(
            "Successfully read and printed node from file_index {:?}",
            file_index
        ),
        Err(e) => println!("Failed to read node: {}", e),
    }
}

macro_rules! define_prob_cache_items {
    ($($variant:ident = $type:ty),+ $(,)?) => {
        #[derive(Clone)]
        pub enum ProbCacheItem {
            $($variant(*mut ProbLazyItem<$type>)),+
        }

        #[derive(Clone, PartialEq, Eq, Hash)]
        pub enum AllocItem {
            $($variant(*mut ProbLazyItem<$type>)),+
        }

        #[derive(Clone, PartialEq, Eq, Hash)]
        pub enum AllocItemState {
            $($variant(*mut ProbLazyItemState<$type>)),+
        }

        pub trait ProbCacheable: Clone + 'static {
            fn from_cache_item(cache_item: ProbCacheItem) -> Option<*mut ProbLazyItem<Self>>;
            fn into_cache_item(item: *mut ProbLazyItem<Self>) -> ProbCacheItem;
        }

        pub trait Allocate: Sized {
            fn from_alloc_item(alloc_item: AllocItem) -> Option<*mut ProbLazyItem<Self>>;
            fn from_alloc_item_state(alloc_item_state: AllocItemState) -> Option<*mut ProbLazyItemState<Self>>;

            fn into_alloc_item(item: *mut ProbLazyItem<Self>) -> AllocItem;
            fn into_alloc_item_state(item: *mut ProbLazyItemState<Self>) -> AllocItemState;
        }

        impl AllocItem {
            fn drop(&self) {
                println!("dropping item");
                unsafe {
                    match self {
                        $(Self::$variant(item) => if !item.is_null() { drop(Box::from_raw(*item)) }),+
                    }
                }
            }
        }

        impl AllocItemState {
            fn drop(&self) {
                println!("dropping state");
                unsafe {
                    match self {
                        $(Self::$variant(state) => if !state.is_null() { drop(Box::from_raw(*state)) }),+
                    }
                }
            }
        }

        $(
            impl ProbCacheable for $type {
                fn from_cache_item(cache_item: ProbCacheItem) -> Option<*mut ProbLazyItem<Self>> {
                    if let ProbCacheItem::$variant(item) = cache_item {
                        Some(item)
                    } else {
                        None
                    }
                }

                fn into_cache_item(item: *mut ProbLazyItem<Self>) -> ProbCacheItem {
                    ProbCacheItem::$variant(item)
                }
            }
        )+

        $(
            impl Allocate for $type {
                fn from_alloc_item(alloc_item: AllocItem) -> Option<*mut ProbLazyItem<Self>> {
                    if let AllocItem::$variant(item) = alloc_item {
                        Some(item)
                    } else {
                        None
                    }
                }

                fn into_alloc_item(item: *mut ProbLazyItem<Self>) -> AllocItem {
                    AllocItem::$variant(item)
                }

                fn from_alloc_item_state(alloc_item_state: AllocItemState) -> Option<*mut ProbLazyItemState<Self>> {
                    if let AllocItemState::$variant(state) = alloc_item_state {
                        Some(state)
                    } else {
                        None
                    }
                }

                fn into_alloc_item_state(state: *mut ProbLazyItemState<Self>) -> AllocItemState {
                    AllocItemState::$variant(state)
                }
            }
        )+
    };
}

define_prob_cache_items! {
    ProbNode = ProbNode,
}

pub struct Allocator {
    states: DashSet<AllocItemState>,
    items: DashSet<AllocItem>,
}

impl Allocator {
    pub fn new() -> Self {
        Self {
            states: DashSet::new(),
            items: DashSet::new(),
        }
    }

    pub fn alloc_state<T: Allocate>(
        &self,
        state: ProbLazyItemState<T>,
    ) -> *mut ProbLazyItemState<T> {
        let ptr = Box::into_raw(Box::new(state));
        self.states.insert(T::into_alloc_item_state(ptr));
        ptr
    }

    pub fn alloc_item<T: Allocate>(&self, item: ProbLazyItem<T>) -> *mut ProbLazyItem<T> {
        let ptr = Box::into_raw(Box::new(item));
        self.items.insert(T::into_alloc_item(ptr));
        ptr
    }

    pub fn free_state<T: Allocate>(&self, state: *mut ProbLazyItemState<T>) -> bool {
        if self
            .states
            .remove(&T::into_alloc_item_state(state))
            .is_some()
        {
            println!("dropping state from allocator");
            unsafe { drop(Box::from_raw(state)) };
            true
        } else {
            false
        }
    }

    pub fn free_item<T: Allocate>(&self, item: *mut ProbLazyItem<T>) -> bool {
        if self.items.remove(&T::into_alloc_item(item)).is_some() {
            println!("dropping item from allocator");
            unsafe { drop(Box::from_raw(item)) };
            true
        } else {
            false
        }
    }

    pub fn get_state<T: Allocate>(
        &self,
        state: *mut ProbLazyItemState<T>,
    ) -> Option<ProbLazyItemState<T>> {
        if self
            .states
            .remove(&T::into_alloc_item_state(state))
            .is_some()
        {
            Some(unsafe { ptr::read(state) })
        } else {
            None
        }
    }
}

unsafe impl Send for Allocator {}
unsafe impl Sync for Allocator {}

impl Drop for Allocator {
    fn drop(&mut self) {
        for item in self.items.iter() {
            item.drop();
        }

        for state in self.states.iter() {
            state.drop();
        }
    }
}

pub struct ProbCache {
    allocator: Arc<Allocator>,
    cuckoo_filter: RwLock<CuckooFilter<u64>>,
    registry: LRUCache<u64, ProbCacheItem>,
    bufmans: Arc<BufferManagerFactory>,
}

impl ProbCache {
    pub fn new(cuckoo_filter_capacity: usize, bufmans: Arc<BufferManagerFactory>) -> Self {
        let allocator = Arc::new(Allocator::new());
        let cuckoo_filter = CuckooFilter::new(cuckoo_filter_capacity);
        let registry = LRUCache::with_prob_eviction(1000, 0.03125);

        Self {
            allocator,
            cuckoo_filter: RwLock::new(cuckoo_filter),
            registry,
            bufmans,
        }
    }

    pub fn get_allocator(&self) -> Arc<Allocator> {
        self.allocator.clone()
    }

    pub fn get_lazy_object<T: ProbCacheable + UpdateSerialized + ProbSerialize + Allocate>(
        self: Arc<Self>,
        file_index: FileIndex,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<*mut ProbLazyItem<T>, BufIoError> {
        let combined_index = Self::combine_index(&file_index);

        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            println!("Acquired read lock on cuckoo_filter");

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&combined_index) {
                println!("FileIndex found in cuckoo_filter");
                if let Some(obj) = self.registry.get(&combined_index) {
                    if let Some(item) = T::from_cache_item(obj) {
                        return Ok(item);
                    }
                } else {
                    println!("Object not found in registry despite being in cuckoo_filter");
                }
            } else {
                println!("FileIndex not found in cuckoo_filter");
            }
        }

        let node = ProbLazyItem::new_pending(&self.allocator, file_index);
        let cache_item = T::into_cache_item(node);

        {
            self.cuckoo_filter.write().unwrap().insert(&combined_index);
            self.registry.insert(combined_index, cache_item);
        }

        let (offset, version_number, version_id) = if let FileIndex::Valid {
            offset,
            version_number,
            version_id,
        } = &file_index
        {
            (*offset, *version_number, *version_id)
        } else {
            (FileOffset(0), 0, 0.into())
        };

        if max_loads == 0 || skipm.contains(&combined_index) {
            return Ok(node);
        }

        skipm.insert(combined_index);

        let (data, versions) = lazy_item_deserialize_impl(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            max_loads - 1,
            skipm,
        )?;
        let data = Arc::new(data);
        let new_state = self.allocator.alloc_state(ProbLazyItemState::Ready {
            data,
            file_offset: Cell::new(Some(offset)),
            decay_counter: 0,
            persist_flag: AtomicBool::new(false),
            serialized_flag: AtomicBool::new(true),
            version_id,
            version_number,
            versions,
        });

        let old_state = unsafe { &*node }.swap_state(new_state);

        self.allocator.free_state(old_state);

        Ok(node)
    }

    pub fn get_object<T: ProbCacheable + UpdateSerialized + ProbSerialize + Allocate>(
        self: Arc<Self>,
        file_index: FileIndex,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<(Arc<T>, ProbLazyItemArray<T, 4>), BufIoError> {
        let combined_index = Self::combine_index(&file_index);

        let (node, is_new) = {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            println!("Acquired read lock on cuckoo_filter");

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&combined_index) {
                println!("FileIndex found in cuckoo_filter");
                if let Some(obj) = self.registry.get(&combined_index) {
                    if let Some(item) = T::from_cache_item(obj) {
                        unsafe {
                            if let ProbLazyItemState::Ready { data, versions, .. } =
                                &*(*item).get_state().load(Ordering::SeqCst)
                            {
                                return Ok((data.clone(), versions.clone()));
                            }
                        }
                        (item, false)
                    } else {
                        (ProbLazyItem::new_pending(&self.allocator, file_index), true)
                    }
                } else {
                    println!("Object not found in registry despite being in cuckoo_filter");
                    (ProbLazyItem::new_pending(&self.allocator, file_index), true)
                }
            } else {
                println!("FileIndex not found in cuckoo_filter");
                (ProbLazyItem::new_pending(&self.allocator, file_index), true)
            }
        };

        let cache_item = T::into_cache_item(node);

        if is_new {
            self.cuckoo_filter.write().unwrap().insert(&combined_index);
            self.registry.insert(combined_index, cache_item);
        }

        let (offset, version_number, version_id) = if let FileIndex::Valid {
            offset,
            version_number,
            version_id,
        } = &file_index
        {
            (*offset, *version_number, *version_id)
        } else {
            (FileOffset(0), 0, 0.into())
        };

        skipm.insert(combined_index);

        let (data, versions) = lazy_item_deserialize_impl(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            max_loads - 1,
            skipm,
        )?;
        let data: Arc<T> = Arc::new(data);
        let new_state = self.allocator.alloc_state(ProbLazyItemState::Ready {
            data: data.clone(),
            file_offset: Cell::new(Some(offset)),
            decay_counter: 0,
            persist_flag: AtomicBool::new(false),
            serialized_flag: AtomicBool::new(true),
            version_id,
            version_number,
            versions: versions.clone(),
        });

        let old_state = unsafe { &*node }.swap_state(new_state);

        self.allocator.free_state(old_state);

        Ok((data, versions))
    }

    pub fn combine_index(file_index: &FileIndex) -> u64 {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => ((offset.0 as u64) << 32) | (**version_id as u64),
            FileIndex::Invalid => u64::MAX, // Use max u64 value for Invalid
        }
    }

    pub fn load_item<T: ProbSerialize>(
        self: &Arc<Self>,
        file_index: FileIndex,
    ) -> Result<T, BufIoError> {
        let mut skipm: HashSet<u64> = HashSet::new();

        if file_index == FileIndex::Invalid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )
            .into());
        };

        T::deserialize(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            1000,
            &mut skipm,
        )
    }
}
