use super::chunked_list::LazyItem;
use super::file_persist::*;
use super::types::*;
use dashmap::DashMap;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::RwLock;

struct NodeRegistry {
    cuckoo_filter: RwLock<CuckooFilter<VectorId>>,
    registry: DashMap<VectorId, LazyItem<MergedNode>>,
}

impl NodeRegistry {
    fn new(cuckoo_filter_capacity: usize) -> Self {
        let cuckoo_filter = CuckooFilter::new(cuckoo_filter_capacity);
        let registry = DashMap::new();
        NodeRegistry {
            cuckoo_filter: RwLock::new(cuckoo_filter),
            registry,
        }
    }

    pub fn get_object<F>(&self, key: VectorId, load_function: F) -> LazyItem<MergedNode>
    where
        F: Fn(&VectorId) -> LazyItem<MergedNode>,
    {
        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&key) {
                // Definitive check with hashmap
                if let Some(obj) = self.registry.get(&key) {
                    return obj.clone();
                }
            }
        }

        // Load the object if it's not in the registry
        let obj = load_function(&key);
        self.registry.insert(key.clone(), obj.clone());

        {
            let mut cuckoo_filter = self.cuckoo_filter.write().unwrap();
            cuckoo_filter.insert(&key);
        }

        obj
    }

    pub fn hash_key(key: &VectorId) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

fn load_object_from_file(key: &VectorId) -> LazyItem<MergedNode> {
    match key {
        VectorId::Str(s) => LazyItem::Null, //  format!("Object loaded for {}", s),
        VectorId::Int(i) => LazyItem::Null, //format!("Object loaded for {}", i),
    }
}

pub fn load_cache() {
    use std::fs::OpenOptions;

    let mut file = OpenOptions::new()
        .read(true)
        .open("index.0")
        .expect("failed to open");

    let offset = 0;
    match read_node_from_file(&mut file, offset) {
        Ok(_) => println!("Successfully read and printed node from offset {}", offset),
        Err(e) => println!("Failed to read node: {}", e),
    }
}
