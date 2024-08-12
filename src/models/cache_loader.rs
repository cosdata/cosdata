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
        println!(
            "get_object called with key: {:?}, max_loads: {}",
            key, max_loads
        );

        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            println!("Acquired read lock on cuckoo_filter");

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&key) {
                println!("Key found in cuckoo_filter");
                if let Some(obj) = self.registry.get(&key) {
                    println!("Object found in registry, returning");
                    return Ok(obj.clone());
                } else {
                    println!("Object not found in registry despite being in cuckoo_filter");
                }
            } else {
                println!("Key not found in cuckoo_filter");
            }
        }
        println!("Released read lock on cuckoo_filter");

        if max_loads == 0 || false == skipm.insert(key) {
            println!("Either max_loads hit 0 or loop detected, returning LazyItem with no data");
            return Ok(LazyItem::Valid {
                data: None,
                offset: Item::new(Some(key)),
                decay_counter: 0,
            });
        }

        println!("Calling load_function");
        let obj = load_function(reader, key, self.clone(), max_loads - 1, skipm)?;
        println!("load_function returned successfully");

        if let Some(obj) = self.registry.get(&key) {
            println!("Object found in registry after load, returning");
            return Ok(obj.clone());
        }

        println!("Creating new LazyItem");
        let item = LazyItem::Valid {
            data: Some(Item::new(obj)),
            offset: Item::new(Some(key)),
            decay_counter: 0,
        };

        println!("Inserting key into cuckoo_filter");
        self.cuckoo_filter.write().unwrap().insert(&key);

        println!("Inserting item into registry");
        self.registry.insert(key, item.clone());

        println!("Returning newly created LazyItem");
        Ok(item)
    }

    pub fn load_item<T: CustomSerialize>(
        self: Arc<Self>,
        offset: FileOffset,
    ) -> std::io::Result<T> {
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

    let offset = FileOffset(0);
    let cache = Arc::new(NodeRegistry::new(1000, file));
    match read_node_from_file(offset, cache) {
        Ok(_) => println!("Successfully read and printed node from offset {}", offset.0),
        Err(e) => println!("Failed to read node: {}", e),
    }
}
