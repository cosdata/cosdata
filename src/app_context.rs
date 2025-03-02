use std::sync::Arc;

use crate::args::CosdataArgs;
use crate::config_loader::Config;
use crate::models::common::WaCustomError;
use crate::models::types::{get_app_env, AppEnv};
use crate::models::collection_cache::CollectionCacheManager;
use std::path::Path;
use rayon::ThreadPool;

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub threadpool: ThreadPool,
    pub ain_env: Arc<AppEnv>,
    pub collection_cache: Arc<CollectionCacheManager>,
}

impl AppContext {
    pub fn new(config: Config, args: CosdataArgs) -> Result<Self, WaCustomError> {
        let ain_env = get_app_env(&config, args)?;
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_pool.pool_size)
            .build()
            .expect("Failed to build thread pool");

        let collection_cache = Arc::new(CollectionCacheManager::new(
            Path::new("./collections/").into(),
            config.cache.max_collections,
            config.cache.collection_ttl_secs,
        ));

        Ok(Self {
            config,
            ain_env,
            threadpool,
            collection_cache,
        })
    }
}
