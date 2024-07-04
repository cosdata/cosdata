use super::file_persist::*;
use super::types::*;
use dashmap::DashMap;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::RwLock;

struct NodeRegistry {
    cuckoo_filter: RwLock<CuckooFilter<VectorId>>,
    registry: DashMap<VectorId, NodeRef>,
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

    pub fn get_object<F>(&self, key: VectorId, load_function: F) -> NodeRef
    where
        F: Fn(&VectorId) -> NodeRef,
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

fn load_object_from_file(key: &VectorId) -> Option<NodeRef> {
    match key {
        VectorId::Str(s) => None, //  format!("Object loaded for {}", s),
        VectorId::Int(i) => None, //format!("Object loaded for {}", i),
    }
}
