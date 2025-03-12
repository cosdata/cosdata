use std::sync::Arc;
use dashmap::DashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use half::f16;
use lmdb::{Environment, Database, DatabaseFlags};
use crate::indexes::inverted_index::InvertedIndex;
use crate::models::common::WaCustomError;
use crate::models::types::DenseIndex;
use crate::models::collection::Collection;
use crate::config_loader::{Config, load_config};
use crate::models::meta_persist::load_collections;
use crate::models::lru_cache::{LRUCache, EvictStrategy, ProbEviction};
use std::fs::OpenOptions;
use std::sync::RwLock;
use std::io::Read;

pub struct CollectionCacheEntry {
    pub name: String,
    pub dense_index: Option<Arc<DenseIndex>>,
    pub inverted_index: Option<Arc<InvertedIndex>>,
    pub last_accessed: Instant,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollectionNameKey(u64);

impl CollectionNameKey {
    pub fn new(name: &str) -> Self {
        // Use SipHash to convert the string to a u64
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

impl From<u64> for CollectionNameKey {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl Into<u64> for CollectionNameKey {
    fn into(self) -> u64 {
        self.0
    }
}

pub struct CollectionCacheManager {
    // Use LRUCache with CollectionNameKey instead of DashMap
    cache: Arc<LRUCache<CollectionNameKey, Arc<CollectionCacheEntry>>>,
    collections_path: Arc<Path>,
    cache_ttl: Duration,
    active_operations: Arc<DashMap<String, u32>>,
    name_to_key: Arc<DashMap<String, CollectionNameKey>>,
}

impl CollectionCacheManager {
    pub fn new(collections_path: Arc<Path>, max_collections: usize, cache_ttl_secs: u64) -> Self {
        // Use a simple constructor for f16
        let prob_f16 = half::f16::from_f32(0.1);
        let strategy = EvictStrategy::Probabilistic(ProbEviction::new(prob_f16));

        Self {
            cache: Arc::new(LRUCache::new(max_collections, strategy)),
            collections_path,
            cache_ttl: Duration::from_secs(cache_ttl_secs),
            active_operations: Arc::new(DashMap::new()),
            name_to_key: Arc::new(DashMap::new()),
        }
    }

    fn get_config(&self) -> Result<Config, WaCustomError> {
        load_config()
    }

    pub fn load_collection(&self, name: &str) -> Result<Arc<CollectionCacheEntry>, WaCustomError> {
        // Get or create key for this collection name
        let key = self.name_to_key
            .entry(name.to_string())
            .or_insert_with(|| CollectionNameKey::new(name))
            .clone();

        // Try to get from cache first
        if let Some(entry) = self.cache.get(&key) {
            // Update operation counter
            self.increment_operation_counter(name);
            return Ok(entry);
        }

        // If not in cache, load it
        let collection_path = self.collections_path.join(name);
        if !collection_path.exists() {
            return Err(WaCustomError::NotFound(format!("Collection '{}' not found", name)));
        }

        // Load dense index and/or inverted index as needed
        let dense_index = self.load_dense_index(name)?;
        let inverted_index = self.load_inverted_index(name)?;

        // Create cache entry
        let entry = Arc::new(CollectionCacheEntry {
            name: name.to_string(),
            dense_index,
            inverted_index,
            last_accessed: Instant::now(),
        });

        // Add to cache and initialize operation counter
        self.cache.insert(key, entry.clone());
        self.increment_operation_counter(name);

        // The LRUCache will handle eviction automatically when capacity is exceeded

        Ok(entry)
    }

    /// Explicitly unload a collection from memory
    pub fn unload_collection(&self, name: &str) -> Result<(), WaCustomError> {
        // Check if collection is currently in use
        if let Some(counter) = self.active_operations.get(name) {
            if *counter > 0 {
                return Err(WaCustomError::DatabaseError(
                    format!("Collection '{}' is currently in use and cannot be unloaded", name)
                ));
            }
        }

        // Get the key for this collection
        let key = match self.name_to_key.get(name) {
            Some(key) => key.clone(),
            None => return Err(WaCustomError::NotFound(format!("Collection '{}' not loaded", name)))
        };

        // We can't directly check access time since LRUCache doesn't expose that
        // So we'll rely only on the active operations check

        // Remove from mappings
        self.name_to_key.remove(name);
        self.active_operations.remove(name);

        // Note: The entry will remain in the LRUCache until evicted

        Ok(())
    }


    fn load_dense_index(&self, name: &str) -> Result<Option<Arc<DenseIndex>>, WaCustomError> {
        log::info!("Loading dense index for collection: {}", name);

        // TODO: In the next implementation ,
        // 1. Check collection configuration for dense vector settings
        // 2. Initialize the appropriate dense index if enabled
        // 3. Configure index with collection-specific parameters

        // Currently returning None as index loading will be implemented
        // as part of the complete collection storage.
        Ok(None)
    }

    fn load_inverted_index(&self, name: &str) -> Result<Option<Arc<InvertedIndex>>, WaCustomError> {
        log::info!("Loading inverted index for collection: {}", name);

        // TODO: In the next implementation , this will retrieve the
        // appropriate inverted index configuration from collection metadata

        // Currently returning None as index loading will be implemented
        // as part of the complete collection storage
        Ok(None)
    }

    /// Get all currently loaded collections
    pub fn get_loaded_collections(&self) -> Vec<String> {
        // Use the name_to_key mapping to determine which collections are loaded
        self.name_to_key.iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Check if a collection is loaded
    pub fn is_loaded(&self, name: &str) -> bool {
        if let Some(key_ref) = self.name_to_key.get(name) {
            // Get the &CollectionNameKey from the Ref<String, CollectionNameKey>
            let key = key_ref.value();
            self.cache.get(key).is_some()
        } else {
            false
        }
    }

    /// Increment the operation counter for a collection
    fn increment_operation_counter(&self, collection_name: &str) {
        self.active_operations
            .entry(collection_name.to_string())
            .and_modify(|counter| *counter += 1)
            .or_insert(1);
    }

    /// Decrement the operation counter for a collection
    pub fn decrement_operation_counter(&self, collection_name: &str) {
        if let Some(mut entry) = self.active_operations.get_mut(collection_name) {
            if *entry > 0 {
                *entry -= 1;
            }
        }
    }

    /// Get the number of active operations for a collection
    pub fn get_active_operations(&self, collection_name: &str) -> u32 {
        self.active_operations
            .get(collection_name)
            .map(|counter| *counter)
            .unwrap_or(0)
    }
}

// Implement a guard to automatically decrement the operation counter when dropped
pub struct CollectionOperationGuard {
    collection_name: String,
    cache_manager: Arc<CollectionCacheManager>,
}

impl CollectionOperationGuard {
    pub fn new(collection_name: String, cache_manager: Arc<CollectionCacheManager>) -> Self {
        // Increment operation counter when guard is created
        cache_manager.increment_operation_counter(&collection_name);
        Self {
            collection_name,
            cache_manager,
        }
    }
}

impl Drop for CollectionOperationGuard {
    fn drop(&mut self) {
        self.cache_manager.decrement_operation_counter(&self.collection_name);
    }
}
