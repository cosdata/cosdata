use std::sync::Arc;

use crate::args::CosdataArgs;
use crate::config_loader::Config;
use crate::models::collection_cache::CollectionCacheManager;
use crate::models::common::WaCustomError;
use crate::models::paths::get_data_path;
use crate::models::types::{get_app_env, AppEnv};
use rayon::ThreadPool;

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub threadpool: ThreadPool,
    pub ain_env: Arc<AppEnv>,
    pub collection_cache_manager: Arc<CollectionCacheManager>,
}

impl AppContext {
    pub fn new(config: Config, args: CosdataArgs) -> Result<Self, WaCustomError> {
        let ain_env = get_app_env(&config, args)?;
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_pool.pool_size)
            .build()
            .expect("Failed to build thread pool");

        let collections_path = get_data_path().join("collections");
        std::fs::create_dir_all(&collections_path)
            .map_err(|e| WaCustomError::FsError(e.to_string()))?;

        let collection_cache_manager = Arc::new(CollectionCacheManager::new(
            Arc::from(collections_path),
            config.cache.max_collections,
            config.cache.eviction_probability,
            ain_env.clone(),
        ));

        Ok(Self {
            config,
            ain_env,
            threadpool,
            collection_cache_manager,
        })
    }
}

use crate::models::collection_cache::CollectionCacheExt;

impl CollectionCacheExt for AppContext {
    fn update_collection_for_transaction(&self, name: &str) -> Result<(), WaCustomError> {
        self.collection_cache_manager.update_collection_usage(name)
    }

    fn update_collection_for_query(&self, name: &str) -> Result<bool, WaCustomError> {
        self.collection_cache_manager
            .probabilistic_update(name, 0.01)
    }
}
