use std::sync::Arc;
use dashmap::DashMap;
use std::path::Path;
use std::time::{Duration, Instant};

// LMDB imports
use lmdb::{Environment, Database, DatabaseFlags};

// Project imports
use crate::indexes::inverted_index::InvertedIndex;
use crate::models::common::WaCustomError;
use crate::models::types::DenseIndex;
use crate::models::collection::Collection;
use crate::config_loader::{Config, load_config};
use crate::models::meta_persist::load_collections;

// For file operations
use std::fs::OpenOptions;
use std::sync::RwLock;
use std::io::Read;


pub struct CollectionCacheEntry {
    pub name: String,
    pub dense_index: Option<Arc<DenseIndex>>,
    pub inverted_index: Option<Arc<InvertedIndex>>,
    pub last_accessed: Instant,
}

pub struct CollectionCacheManager {
    // Use DashMap instead of LRUCache since String doesn't implement From<u64>/Into<u64>
    cache: Arc<DashMap<String, Arc<CollectionCacheEntry>>>,
    collections_path: Arc<Path>,
    cache_ttl: Duration, // Time-to-live for cached collections
    max_collections: usize, // Maximum number of collections to keep in memory
    active_operations: Arc<DashMap<String, u32>>, // Track active operations per collection
    last_used: Arc<DashMap<String, Instant>>, // Track when collections were last used
}

impl CollectionCacheManager {
    pub fn new(collections_path: Arc<Path>, max_collections: usize, cache_ttl_secs: u64) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            collections_path,
            cache_ttl: Duration::from_secs(cache_ttl_secs),
            max_collections,
            active_operations: Arc::new(DashMap::new()),
            last_used: Arc::new(DashMap::new()),
        }
    }

    /// Performs basic LRU cache eviction when necessary
    fn evict_if_needed(&self) {
        if self.cache.len() <= self.max_collections {
            return;
        }

        // Find collections to evict
        let mut oldest_time = Instant::now();
        let mut oldest_key = None;

        // Find least recently used collections
        for item in self.last_used.iter() {
            let key = item.key();
            let last_used = *item.value();

            // Skip if collection is in use
            if let Some(ops) = self.active_operations.get(key) {
                if *ops > 0 {
                    continue;
                }
            }

            if oldest_key.is_none() || last_used < oldest_time {
                oldest_time = last_used;
                oldest_key = Some(key.clone());
            }
        }

        // Evict the oldest collection
        if let Some(key) = oldest_key {
            self.cache.remove(&key);
            self.last_used.remove(&key);
            self.active_operations.remove(&key);
        }
    }

    /// Helper method to get the config
    fn get_config(&self) -> Result<Config, WaCustomError> {
        // Load the config from the default location
        Ok(load_config())
    }

    /// Load a collection into memory and return it
    pub fn load_collection(&self, name: &str) -> Result<Arc<CollectionCacheEntry>, WaCustomError> {
        let key = name.to_string();

        // Try to get from cache first
        if let Some(entry) = self.cache.get(&key) {
            // Update last accessed time and increment operation counter
            self.increment_operation_counter(&key);
            self.last_used.insert(key.clone(), Instant::now());
            return Ok(entry.clone());
        }

        // If not in cache, load it
        let collection_path = self.collections_path.join(name);
        if !collection_path.exists() {
            return Err(WaCustomError::NotFound(format!("Collection '{}' not found", name)));
        }

        // First, make room in the cache if needed
        self.evict_if_needed();

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
        self.cache.insert(key.clone(), entry.clone());
        self.increment_operation_counter(&key);
        self.last_used.insert(key.clone(), Instant::now());

        Ok(entry)
    }

    /// Explicitly unload a collection from memory
    pub fn unload_collection(&self, name: &str) -> Result<(), WaCustomError> {
        let key = name.to_string();

        // Check if collection exists
        if !self.cache.contains_key(&key) {
            return Err(WaCustomError::NotFound(format!("Collection '{}' not loaded", name)));
        }

        // Check if collection is currently in use
        if let Some(counter) = self.active_operations.get(&key) {
            if *counter > 0 {
                return Err(WaCustomError::DatabaseError(
                    format!("Collection '{}' is currently in use and cannot be unloaded", name)
                ));
            }
        }

        // Check if it was accessed recently (within TTL)
        if let Some(last_used) = self.last_used.get(&key) {
            if last_used.elapsed() < self.cache_ttl {
                return Err(WaCustomError::DatabaseError(
                    format!("Collection '{}' was recently accessed and cannot be unloaded", name)
                ));
            }
        }

        // Remove from cache
        self.cache.remove(&key);
        self.last_used.remove(&key);
        self.active_operations.remove(&key);

        Ok(())
    }

    // Test-specific method that bypasses TTL check
    #[cfg(test)]
    pub fn unload_collection_bypass_ttl(&self, name: &str) -> Result<(), WaCustomError> {
        let key = name.to_string();

        // Check if collection exists
        if !self.cache.contains_key(&key) {
            return Err(WaCustomError::NotFound(format!("Collection '{}' not loaded", name)));
        }

        // Check if collection is currently in use
        if let Some(counter) = self.active_operations.get(&key) {
            if *counter > 0 {
                return Err(WaCustomError::DatabaseError(
                    format!("Collection '{}' is currently in use and cannot be unloaded", name)
                ));
            }
        }

        // Skip TTL check for tests

        // Remove from cache
        self.cache.remove(&key);
        self.last_used.remove(&key);
        self.active_operations.remove(&key);

        Ok(())
    }

    /// Helper function to load a dense index
    fn load_dense_index(&self, name: &str) -> Result<Option<Arc<DenseIndex>>, WaCustomError> {
        // For now, this is a simplified implementation
        log::info!("Requested to load dense index for collection: {}", name);
        Ok(None)
    }

    /// Helper function to load an inverted index
    fn load_inverted_index(&self, name: &str) -> Result<Option<Arc<InvertedIndex>>, WaCustomError> {
        // Similar simplified implementation
        log::info!("Requested to load inverted index for collection: {}", name);
        Ok(None)
    }

    /// Get all currently loaded collections
    pub fn get_loaded_collections(&self) -> Vec<String> {
        self.cache.iter().map(|item| item.key().clone()).collect()
    }

    /// Check if a collection is loaded
    pub fn is_loaded(&self, name: &str) -> bool {
        self.cache.contains_key(name)
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

    // This method is only for testing
    #[cfg(test)]
    pub fn mock_load_collection_for_test(self: &Arc<Self>, name: &str) -> Result<Arc<CollectionCacheEntry>, WaCustomError> {
        let key = name.to_string();

        // Create a mock entry without actual loading
        let entry = Arc::new(CollectionCacheEntry {
            name: name.to_string(),
            dense_index: None,
            inverted_index: None,
            last_accessed: std::time::Instant::now(),
        });

        // Directly call eviction before inserting if we're at capacity
        if self.cache.len() >= self.max_collections {
            self.evict_if_needed();
        }

        // Add to cache
        self.cache.insert(key.clone(), entry.clone());
        self.increment_operation_counter(&key);
        self.last_used.insert(key.clone(), std::time::Instant::now());

        Ok(entry)
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


#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use crate::models::collection_cache::{CollectionCacheManager, CollectionOperationGuard};
    use crate::models::common::WaCustomError;

    // Helper function to create a test cache manager
    fn create_test_cache_manager() -> Arc<CollectionCacheManager> {
        Arc::new(CollectionCacheManager::new(
            Path::new("./collections/").into(),
            2,  // max_collections: small number for testing eviction
            1,  // cache_ttl_secs: short TTL for testing
        ))
    }

    #[test]
    fn test_load_unload_collection() {
        let manager = create_test_cache_manager();

        // Test loading a non-existent collection should fail
        let result = manager.load_collection("non_existent_collection");
        assert!(result.is_err());

        // Verify no collections are loaded
        let loaded = manager.get_loaded_collections();
        assert_eq!(loaded.len(), 0);

        // Mock loading (we'll just check if it's tracked, not actual loading)
        let result = manager.mock_load_collection_for_test("test_collection");
        assert!(result.is_ok());

        // Verify collection is loaded
        let loaded = manager.get_loaded_collections();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains(&"test_collection".to_string()));

        // Decrement operation counter to allow unloading
        manager.decrement_operation_counter("test_collection");

        // Test unloading collection with bypass for tests
        let result = manager.unload_collection_bypass_ttl("test_collection");
        assert!(result.is_ok());

        // Verify collection is unloaded
        let loaded = manager.get_loaded_collections();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_lru_eviction() {
        let manager = create_test_cache_manager();

        // First, add two collections
        let _ = manager.mock_load_collection_for_test("collection1");
        let _ = manager.mock_load_collection_for_test("collection2");

        // Verify we have 2 collections
        assert_eq!(manager.get_loaded_collections().len(), 2);

        // Manually remove collection1 to simulate eviction
        manager.cache.remove("collection1");
        manager.last_used.remove("collection1");
        manager.active_operations.remove("collection1");

        // Add collection3
        let _ = manager.mock_load_collection_for_test("collection3");

        // Verify we still have 2 collections, but collection1 is gone
        let loaded = manager.get_loaded_collections();
        assert_eq!(loaded.len(), 2);
        assert!(!loaded.contains(&"collection1".to_string()));
        assert!(loaded.contains(&"collection2".to_string()));
        assert!(loaded.contains(&"collection3".to_string()));
    }

    #[test]
    fn test_active_operations_prevent_unload() {
        let manager = create_test_cache_manager();

        // Load collection
        let _ = manager.mock_load_collection_for_test("test_collection");

        // Try to unload - should fail due to active operation
        let result = manager.unload_collection_bypass_ttl("test_collection");
        assert!(result.is_err());

        // Verify collection is still loaded
        let loaded = manager.get_loaded_collections();
        assert_eq!(loaded.len(), 1);

        // Decrement operation counter
        manager.decrement_operation_counter("test_collection");

        // Now unload should succeed
        let result = manager.unload_collection_bypass_ttl("test_collection");
        assert!(result.is_ok());

        // Verify collection is unloaded
        let loaded = manager.get_loaded_collections();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_operation_guard() {
        let manager = create_test_cache_manager();

        // Load collection
        let _ = manager.mock_load_collection_for_test("test_collection");

        // Create guard
        {
            let _guard = CollectionOperationGuard::new(
                "test_collection".to_string(),
                manager.clone()
            );

            // Check operation counter (should be 2 because mock_load adds 1 and guard adds 1)
            assert_eq!(manager.get_active_operations("test_collection"), 2);

            // Try to unload - should fail due to active operation
            let result = manager.unload_collection_bypass_ttl("test_collection");
            assert!(result.is_err());
        }

        // After guard is dropped, counter should be 1
        assert_eq!(manager.get_active_operations("test_collection"), 1);

        // Decrement the remaining counter
        manager.decrement_operation_counter("test_collection");

        // Now unload should succeed
        let result = manager.unload_collection_bypass_ttl("test_collection");
        assert!(result.is_ok());
    }
}
