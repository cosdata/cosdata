use log::info;
use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::indexes::hnsw::HNSWIndex;
use crate::indexes::inverted::InvertedIndex;
use crate::models::common::WaCustomError;
use crate::models::lru_cache::{EvictStrategy, LRUCache, ProbEviction};
use crate::models::types::AppEnv;

#[allow(dead_code)]
pub struct CollectionCacheEntry {
    pub name: String,
    pub dense_index: Option<Arc<HNSWIndex>>,
    pub inverted_index: Option<Arc<InvertedIndex>>,
    pub last_accessed: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollectionNameKey(u64);

impl CollectionNameKey {
    pub fn new(name: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }
}

impl From<u64> for CollectionNameKey {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<CollectionNameKey> for u64 {
    fn from(key: CollectionNameKey) -> u64 {
        key.0
    }
}

// Manages the caching of collections and their indexes
pub struct CollectionCacheManager {
    cache: Arc<LRUCache<CollectionNameKey, Arc<CollectionCacheEntry>>>,
    collections_path: Arc<Path>,
    name_to_key: Arc<dashmap::DashMap<String, CollectionNameKey>>,
    app_env: Arc<AppEnv>,
}

impl CollectionCacheManager {
    #[allow(unused)]
    pub fn new(
        collections_path: Arc<Path>,
        max_collections: usize,
        eviction_probability: f32,
        app_env: Arc<AppEnv>,
    ) -> Self {
        // Create maps for name-key mapping
        let name_to_key = Arc::new(dashmap::DashMap::new());

        let prob_f16 = half::f16::from_f32(eviction_probability);
        let strategy = EvictStrategy::Probabilistic(ProbEviction::new(prob_f16));

        // Create the LRUCache
        let cache = Arc::new(LRUCache::new(max_collections, strategy));

        // Create the manager instance
        let manager = Self {
            cache: cache.clone(),
            collections_path,
            name_to_key,
            app_env,
        };

        // Set the eviction hook function for logging only
        let cache_mut = unsafe {
            let cache_ptr = Arc::as_ptr(&cache);
            &mut *(cache_ptr as *mut LRUCache<CollectionNameKey, Arc<CollectionCacheEntry>>)
        };

        // logging hook for eviction events
        cache_mut.set_evict_hook(Some(|entry: &Arc<CollectionCacheEntry>| {
            info!("Collection '{}' was evicted from cache", entry.name);
        }));

        manager
    }

    // Gets or creates a key for a collection name
    fn get_or_create_key(&self, name: &str) -> CollectionNameKey {
        match self.name_to_key.get(name) {
            Some(key) => key.clone(),
            None => {
                let new_key = CollectionNameKey::new(name);
                self.name_to_key.insert(name.to_string(), new_key.clone());
                new_key
            }
        }
    }

    pub fn load_collection(&self, name: &str) -> Result<Arc<CollectionCacheEntry>, WaCustomError> {
        // Get or create key for this collection name
        let key = self.get_or_create_key(name);

        // Try to get from cache first - this will update LRU tracking
        if let Some(entry) = self.cache.get(&key) {
            info!("Collection '{}' found in cache", name);
            return Ok(entry);
        }

        // If not in cache, load it
        info!("Loading collection '{}' into cache", name);
        let collection_path = self.collections_path.join(name);
        if !collection_path.exists() {
            return Err(WaCustomError::NotFound(format!(
                "Collection '{}' not found",
                name
            )));
        }

        // Load dense and inverted indexes
        let dense_index = self.load_dense_index(name)?;
        let inverted_index = self.load_inverted_index(name)?;

        // Create cache entry
        let entry = Arc::new(CollectionCacheEntry {
            name: name.to_string(),
            dense_index,
            inverted_index,
            last_accessed: Instant::now(),
        });

        // Add to cache - this may trigger eviction if at capacity
        self.cache.insert(key, entry.clone());
        info!(
            "Added collection '{}' to cache. Current loaded collections count: {}",
            name,
            self.name_to_key.len()
        );

        Ok(entry)
    }

    pub fn unload_collection(&self, name: &str) -> Result<(), WaCustomError> {
        // Clean up mappings
        self.name_to_key.remove(name);

        // Log the unloading
        info!("Explicitly unloaded collection '{}'", name);
        Ok(())
    }

    pub fn update_collection_usage(&self, name: &str) -> Result<(), WaCustomError> {
        let key = match self.name_to_key.get(name) {
            Some(key) => key.clone(),
            None => {
                // If collection is not loaded, load it
                self.load_collection(name)?;
                return Ok(());
            }
        };

        // Get from cache - this updates its recency
        if self.cache.get(&key).is_none() {
            // Collection was evicted but mapping remains - clean up and load
            self.name_to_key.remove(name);
            self.load_collection(name)?;
        }

        Ok(())
    }

    pub fn probabilistic_update(
        &self,
        name: &str,
        probability: f32,
    ) -> Result<bool, WaCustomError> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < probability {
            self.update_collection_usage(name)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // Loads the dense vector index for a collection
    fn load_dense_index(&self, name: &str) -> Result<Option<Arc<HNSWIndex>>, WaCustomError> {
        info!("Loading dense index for collection: {}", name);

        // First check if the index exists in the AppEnv
        if let Some(index) = self
            .app_env
            .collections_map
            .get_collection(name)
            .and_then(|collection| collection.get_hnsw_index())
        {
            info!("Found HNSW index for '{}' in memory", name);
            return Ok(Some(index));
        }

        // If not in memory, we would load it from disk
        info!("No HNSW index found for '{}'", name);
        Ok(None)
    }

    // Loads the inverted index for a collection
    fn load_inverted_index(&self, name: &str) -> Result<Option<Arc<InvertedIndex>>, WaCustomError> {
        info!("Loading inverted index for collection: {}", name);

        // First check if the index exists in the AppEnv
        if let Some(index) = self
            .app_env
            .collections_map
            .get_collection(name)
            .and_then(|collection| collection.get_inverted_index())
        {
            info!("Found inverted index for '{}' in memory", name);
            return Ok(Some(index));
        }

        // If not in memory, same logic as for dense index
        info!("No inverted index found for '{}'", name);
        Ok(None)
    }

    // Gets a list of all currently loaded collections
    pub fn get_loaded_collections(&self) -> Vec<String> {
        self.name_to_key
            .iter()
            .filter_map(|entry| {
                let key = entry.value();
                if self.cache.get(key).is_some() {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect()
    }

    // Checks if a collection is loaded
    #[allow(dead_code)]
    pub fn is_loaded(&self, name: &str) -> bool {
        if let Some(key_ref) = self.name_to_key.get(name) {
            let key = key_ref.value();
            self.cache.get(key).is_some()
        } else {
            false
        }
    }

    // Cleans up any stale mappings that point to collections no longer in the cache
    #[allow(dead_code)]
    pub fn cleanup_stale_mappings(&self) -> Result<usize, WaCustomError> {
        let mut removed_count = 0;

        // Create a list of collection names to avoid mutating during iteration
        let collection_names: Vec<String> = self
            .name_to_key
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        for name in collection_names {
            if let Some(key) = self.name_to_key.get(&name) {
                // Check if the collection is still in the cache
                if self.cache.get(&key.clone()).is_none() {
                    // If not in cache, remove the mappings
                    self.name_to_key.remove(&name);
                    removed_count += 1;
                    info!("Cleaned up stale mapping for collection: {}", name);
                }
            }
        }

        info!(
            "Cleanup completed: removed {} stale mappings",
            removed_count
        );
        Ok(removed_count)
    }
}

// Extension trait for AppContext to easily update collection cache
#[allow(unused)]
pub trait CollectionCacheExt {
    fn update_collection_for_transaction(&self, name: &str) -> Result<(), WaCustomError>;

    fn update_collection_for_query(&self, name: &str) -> Result<bool, WaCustomError>;
}
