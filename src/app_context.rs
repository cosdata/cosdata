use std::path::Path;
use std::sync::Arc;

use crate::config_loader::Config;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::NodeRegistry;
use crate::models::types::{get_app_env, AppEnv};
use crate::WaCustomError;

fn init_index_manager() -> BufferManagerFactory {
    BufferManagerFactory::new(Path::new(".").into(), |root, ver| {
        root.join(format!("{}.index", **ver))
    })
}

fn init_node_registry(index_manager: Arc<BufferManagerFactory>) -> NodeRegistry {
    // @TODO: May be the value can be taken from config
    let cuckoo_filter_capacity = 1000;
    NodeRegistry::new(cuckoo_filter_capacity, index_manager)
}

fn init_vec_raw_manager() -> BufferManagerFactory {
    BufferManagerFactory::new(Path::new(".").into(), |root, ver| {
        root.join(format!("{}.vec_raw", **ver))
    })
}

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub ain_env: Arc<AppEnv>,
}

impl AppContext {
    pub fn new(config: Config) -> Result<Self, WaCustomError> {
        let ain_env = get_app_env()?;
        Ok(Self { config, ain_env })
    }
}
