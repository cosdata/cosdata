use std::sync::Arc;

use crate::config_loader::Config;
use crate::models::types::{get_app_env, AppEnv};
use crate::WaCustomError;
use rayon::ThreadPool;

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub threadpool: ThreadPool,
    pub ain_env: Arc<AppEnv>,
}

impl AppContext {
    pub fn new(config: Config) -> Result<Self, WaCustomError> {
        let ain_env = get_app_env(&config)?;
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_pool.pool_size)
            .build()
            .expect("Failed to build thread pool");

        Ok(Self {
            config,
            ain_env,
            threadpool,
        })
    }
}
