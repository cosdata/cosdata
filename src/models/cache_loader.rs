use super::file_persist::*;
use super::lazy_load::LazyItem;
use super::serializer::CustomSerialize;
use super::types::*;
use dashmap::DashMap;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::collections::HashSet;
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
        skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<LazyItem<MergedNode>>
    where
        F: Fn(
            &mut R,
            FileOffset,
            Arc<Self>,
            u16,
            &mut HashSet<FileOffset>,
        ) -> std::io::Result<MergedNode>,
    {
        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&key) {
                if let Some(obj) = self.registry.get(&key) {
                    println!("Object found in registry, returning");
                    return Ok(obj.clone());
                }
            }
        }

        let mut data = Item::new(MergedNode::new(0, 0));

        {
            let mut cuckoo_filter = self.cuckoo_filter.write().unwrap();

            cuckoo_filter.insert(&key);

            let dummy_data = LazyItem::Valid {
                data: Some(data.clone()),
                offset: Item::new(Some(key)),
                decay_counter: 0,
            };
            self.registry.insert(key, dummy_data);
        }

        if max_loads == 0 || false == skipm.insert(key) {
            println!("max_loads is 0, returning LazyItem with no data");
            return Ok(LazyItem::Valid {
                data: None,
                offset: Item::new(Some(key)),
                decay_counter: 0,
            });
        }

        let obj = match load_function(reader, key, self.clone(), max_loads - 1, skipm) {
            Ok(obj) => obj,
            Err(err) => {
                let mut cuckoo_filter = self.cuckoo_filter.write().unwrap();
                cuckoo_filter.remove(&key);
                self.registry.remove(&key);
                return Err(err);
            }
        };

        data.update(obj.clone());
        Ok(self.registry.get(&key).unwrap().clone())
    }

    pub fn load_item<T: CustomSerialize>(self: Arc<Self>, offset: u32) -> std::io::Result<T> {
        let mut reader_lock = self.reader.write().unwrap();
        let mut skipm: HashSet<FileOffset> = HashSet::new();
        T::deserialize(&mut *reader_lock, offset, self.clone(), 1000, &mut skipm)
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
