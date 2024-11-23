use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::file_persist::read_prop_from_file;
use super::lazy_load::{FileIndex, LazyItem, LazyItemVec, VectorData};
use super::lru_cache::LRUCache;
use super::prob_lazy_load::lazy_item::{ProbLazyItem, ProbLazyItemState};
use super::prob_node::ProbNode;
use super::serializer::prob::{ProbSerialize, UpdateSerialized};
use super::serializer::CustomSerialize;
use super::types::*;
use super::versioning::Hash;
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
use dashmap::DashMap;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::cell::Cell;
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::sync::Weak;
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

macro_rules! define_prob_cache_items {
    ($($variant:ident = $type:ty),+ $(,)?) => {
        #[derive(Clone)]
        pub enum ProbCacheItem {
            $($variant(Arc<ProbLazyItem<$type>>)),+
        }

        pub trait ProbCacheable:  'static {
            fn from_cache_item(cache_item: ProbCacheItem) -> Option<Arc<ProbLazyItem<Self>>>;
            fn into_cache_item(item: Arc<ProbLazyItem<Self>>) -> ProbCacheItem;
        }

        $(
            impl ProbCacheable for $type {
                fn from_cache_item(cache_item: ProbCacheItem) -> Option<Arc<ProbLazyItem<Self>>> {
                    #[allow(irrefutable_let_patterns)]
                    if let ProbCacheItem::$variant(item) = cache_item {
                        Some(item)
                    } else {
                        None
                    }
                }

                fn into_cache_item(item: Arc<ProbLazyItem<Self>>) -> ProbCacheItem {
                    ProbCacheItem::$variant(item)
                }
            }
        )+
    };
}

define_prob_cache_items! {
    ProbNode = ProbNode,
}

pub struct ProbCache {
    cuckoo_filter: RwLock<CuckooFilter<u64>>,
    registry: LRUCache<u64, ProbCacheItem>,
    props_registry: DashMap<u64, Weak<NodeProp>>,
    bufmans: Arc<BufferManagerFactory>,
    prop_file: Arc<File>,
}

impl ProbCache {
    pub fn new(
        cuckoo_filter_capacity: usize,
        bufmans: Arc<BufferManagerFactory>,
        prop_file: Arc<File>,
    ) -> Self {
        let cuckoo_filter = CuckooFilter::new(cuckoo_filter_capacity);
        let registry = LRUCache::with_prob_eviction(1000, 0.03125);
        let props_registry = DashMap::new();

        Self {
            cuckoo_filter: RwLock::new(cuckoo_filter),
            registry,
            props_registry,
            bufmans,
            prop_file,
        }
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
        let prop = Arc::new(read_prop_from_file((offset, length), &self.prop_file)?);
        let weak = Arc::downgrade(&prop);
        self.props_registry.insert(key, weak);
        Ok(prop)
    }

    pub fn insert_lazy_object(
        &self,
        version: Hash,
        offset: u32,
        item: Arc<ProbLazyItem<ProbNode>>,
    ) {
        let combined_index = (offset as u64) << 32 | (*version as u64);
        let mut cuckoo_filter = self.cuckoo_filter.write().unwrap();
        cuckoo_filter.insert(&combined_index);
        if let Some(node) = item.get_lazy_data() {
            let prop_key = Self::get_prop_key(node.prop.location.0, node.prop.location.1);
            self.props_registry
                .insert(prop_key, Arc::downgrade(&node.prop));
        }
        self.registry
            .insert(combined_index, ProbCacheItem::ProbNode(item));
    }

    pub fn get_lazy_object<T: ProbCacheable + UpdateSerialized + ProbSerialize>(
        self: Arc<Self>,
        file_index: FileIndex,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Arc<ProbLazyItem<T>>, BufIoError> {
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

        let node = ProbLazyItem::new_pending(file_index);
        let cache_item = T::into_cache_item(node.clone());

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

        let data = Arc::new(T::deserialize(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            max_loads - 1,
            skipm,
        )?);
        let new_state = Arc::new(ProbLazyItemState::Ready {
            data,
            file_offset: Cell::new(Some(offset)),
            decay_counter: 0,
            persist_flag: AtomicBool::new(false),
            version_id,
            version_number,
        });

        node.set_state(new_state);

        Ok(node)
    }

    pub fn get_object<T: ProbCacheable + UpdateSerialized + ProbSerialize>(
        self: Arc<Self>,
        file_index: FileIndex,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Arc<T>, BufIoError> {
        println!("get_object is called");
        let combined_index = Self::combine_index(&file_index);

        let (node, is_new) = {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            println!("Acquired read lock on cuckoo_filter");

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&combined_index) {
                println!("FileIndex found in cuckoo_filter");
                if let Some(obj) = self.registry.get(&combined_index) {
                    if let Some(item) = T::from_cache_item(obj) {
                        if let ProbLazyItemState::Ready { data, .. } = &*item.get_state() {
                            return Ok(data.clone());
                        }

                        (item, false)
                    } else {
                        (ProbLazyItem::new_pending(file_index), true)
                    }
                } else {
                    println!("Object not found in registry despite being in cuckoo_filter");
                    (ProbLazyItem::new_pending(file_index), true)
                }
            } else {
                println!("FileIndex not found in cuckoo_filter");
                (ProbLazyItem::new_pending(file_index), true)
            }
        };

        let cache_item = T::into_cache_item(node.clone());

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

        let data = Arc::new(T::deserialize(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            max_loads - 1,
            skipm,
        )?);
        let new_state = Arc::new(ProbLazyItemState::Ready {
            data: data.clone(),
            file_offset: Cell::new(Some(offset)),
            decay_counter: 0,
            persist_flag: AtomicBool::new(false),
            version_id,
            version_number,
        });

        node.set_state(new_state);

        Ok(data)
    }

    pub fn combine_index(file_index: &FileIndex) -> u64 {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => ((offset.0 as u64) << 32) | (**version_id as u64),
            FileIndex::Invalid => u64::MAX, // Use max u64 value for Invalid
        }
    }

    pub fn get_prop_key(
        FileOffset(file_offset): FileOffset,
        BytesToRead(length): BytesToRead,
    ) -> u64 {
        (file_offset as u64) << 32 | (length as u64)
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
