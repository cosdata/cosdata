use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::file_persist::*;
use super::lazy_load::{FileIndex, LazyItem, LazyItemVec, VectorData};
use super::lru_cache::LRUCache;
use super::serializer::CustomSerialize;
use super::types::*;
use crate::models::lru_cache::CachedValue;
use crate::storage::inverted_index_old::InvertedIndexItem;
use crate::storage::inverted_index_sparse_ann::{
    InvertedIndexSparseAnn, InvertedIndexSparseAnnNode,
};
use crate::storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasic;
use crate::storage::inverted_index_sparse_ann_new_ds::InvertedIndexNewDSNode;
use crate::storage::Storage;
use arcshift::ArcShift;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::sync::{atomic::AtomicBool, Arc, RwLock};

#[derive(Clone)]
pub enum CacheItem {
    MergedNode(LazyItem<MergedNode>),
    Storage(LazyItem<Storage>),
    InvertedIndexItemWithStorage(LazyItem<InvertedIndexItem<Storage>>),
    Float(LazyItem<f32>),
    Unsigned32(LazyItem<u32>),
    InvertedIndexItemWithFloat(LazyItem<InvertedIndexItem<f32>>),
    InvertedIndexSparseAnnNode(LazyItem<InvertedIndexSparseAnnNode>),
    InvertedIndexSparseAnnNodeBasic(LazyItem<InvertedIndexSparseAnnNodeBasic>),
    InvertedIndexSparseAnn(LazyItem<InvertedIndexSparseAnn>),
    InvertedIndexNewDSNode(LazyItem<InvertedIndexNewDSNode>),
    VectorData(LazyItem<STM<VectorData>>),
}

pub trait Cacheable: Clone + 'static {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>>;
    fn into_cache_item(item: LazyItem<Self>) -> CacheItem;
}

impl Cacheable for MergedNode {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::MergedNode(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::MergedNode(item)
    }
}

impl Cacheable for Storage {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::Storage(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::Storage(item)
    }
}

impl Cacheable for InvertedIndexItem<Storage> {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::InvertedIndexItemWithStorage(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::InvertedIndexItemWithStorage(item)
    }
}

impl Cacheable for f32 {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::Float(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::Float(item)
    }
}

impl Cacheable for u32 {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::Unsigned32(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::Unsigned32(item)
    }
}

impl Cacheable for InvertedIndexItem<f32> {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::InvertedIndexItemWithFloat(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::InvertedIndexItemWithFloat(item)
    }
}

impl Cacheable for InvertedIndexSparseAnnNode {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::InvertedIndexSparseAnnNode(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::InvertedIndexSparseAnnNode(item)
    }
}

impl Cacheable for InvertedIndexSparseAnn {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::InvertedIndexSparseAnn(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::InvertedIndexSparseAnn(item)
    }
}

impl Cacheable for InvertedIndexSparseAnnNodeBasic {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::InvertedIndexSparseAnnNodeBasic(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::InvertedIndexSparseAnnNodeBasic(item)
    }
}

impl Cacheable for InvertedIndexNewDSNode {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::InvertedIndexNewDSNode(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::InvertedIndexNewDSNode(item)
    }
}

impl Cacheable for STM<VectorData> {
    fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
        if let CacheItem::VectorData(item) = cache_item {
            Some(item)
        } else {
            None
        }
    }

    fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
        CacheItem::VectorData(item)
    }
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
