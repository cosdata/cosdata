use std::path::Path;
use std::sync::Arc;

use crate::config_loader::Config;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::NodeRegistry;
use crate::models::types::{get_app_env, AppEnv};

fn init_index_manager() -> BufferManagerFactory {
    BufferManagerFactory::new(
        Path::new(".").into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    )
}

fn init_node_registry(index_manager: Arc<BufferManagerFactory>) -> NodeRegistry {
    // @TODO: May be the value can be taken from config
    let cuckoo_filter_capacity = 1000;
    NodeRegistry::new(cuckoo_filter_capacity, index_manager)
}

fn init_vec_raw_manager() -> BufferManagerFactory {
    BufferManagerFactory::new(
        Path::new(".").into(),
        |root, ver| root.join(format!("{}.vec_raw", **ver))
    )
}

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub node_registry: Arc<NodeRegistry>,
    pub index_manager: Arc<BufferManagerFactory>,
    pub vec_raw_manager: Arc<BufferManagerFactory>,
    pub ain_env: Arc<AppEnv>,
}

impl AppContext {

    pub fn new(config: Config) -> Self {
        let index_manager = Arc::new(init_index_manager());
        let node_registry = Arc::new(init_node_registry(index_manager.clone()));
        let vec_raw_manager = Arc::new(init_vec_raw_manager());
        // Let it panic if there's a problem initializing the
        // env. Without app env, the HTTP server won't be able to
        // serve any incoming requests anyway.
        let ain_env = get_app_env().expect("Failed to initialize app env");
        Self {
            config,
            node_registry,
            index_manager,
            vec_raw_manager,
            ain_env,
        }
    }
}

