use super::chunked_list::LazyItem;
use super::file_persist::*;
use super::serializer::CustomSerialize;
use super::types::*;
use dashmap::DashMap;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::io::Seek;
use std::sync::Arc;
use std::sync::RwLock;

pub struct NodeRegistry<R: Read + Seek> {
    cuckoo_filter: RwLock<CuckooFilter<FileOffset>>,
    registry: DashMap<FileOffset, LazyItem<MergedNode>>,
    reader: Arc<RwLock<R>>,
}

impl<R: Read + Seek> NodeRegistry<R> {
    pub fn new(cuckoo_filter_capacity: usize, reader: R) -> Self {
        let cuckoo_filter = CuckooFilter::new(cuckoo_filter_capacity);
        let registry = DashMap::new();
        NodeRegistry {
            cuckoo_filter: RwLock::new(cuckoo_filter),
            registry,
            reader: Arc::new(RwLock::new(reader)),
        }
    }

    pub fn get_object<F>(
        self: Arc<Self>,
        key: FileOffset,
        reader: &mut R,
        load_function: F,
        max_loads: u16,
    ) -> std::io::Result<LazyItem<MergedNode>>
    where
        F: Fn(&mut R, FileOffset, Arc<Self>, u16) -> std::io::Result<MergedNode>,
    {
        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&key) {
                if let Some(obj) = self.registry.get(&key) {
                    return Ok(obj.clone());
                }
            }
        }

        if max_loads == 0 {
            return Ok(LazyItem {
                data: None,
                offset: Some(key),
                decay_counter: 0,
            });
        }

        let obj = load_function(reader, key, self.clone(), max_loads - 1)?;

        if let Some(obj) = self.registry.get(&key) {
            return Ok(obj.clone());
        }

        let item = LazyItem {
            data: Some(Arc::new(RwLock::new(obj))),
            offset: Some(key),
            decay_counter: 0,
        };

        self.cuckoo_filter.write().unwrap().insert(&key);
        self.registry.insert(key, item.clone());

        Ok(item)
    }

    pub fn load_item<T: CustomSerialize>(self: Arc<Self>, offset: u32) -> std::io::Result<T> {
        let mut reader_lock = self.reader.write().unwrap();
        T::deserialize(&mut *reader_lock, offset, self.clone(), 5)
    }

    pub fn hash_key(key: &VectorId) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

pub fn load_cache() {
    use std::fs::OpenOptions;

    let file = OpenOptions::new()
        .read(true)
        .open("0.index")
        .expect("failed to open");

    let offset = 0;
    let cache = Arc::new(NodeRegistry::new(1000, file));
    match read_node_from_file(offset, cache) {
        Ok(_) => println!("Successfully read and printed node from offset {}", offset),
        Err(e) => println!("Failed to read node: {}", e),
    }
}
