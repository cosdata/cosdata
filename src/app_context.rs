use std::path::Path;
use std::sync::Arc;

use crate::config_loader::Config;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::NodeRegistry;

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

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub node_registry: Arc<NodeRegistry>,
    pub index_manager: Arc<BufferManagerFactory>,
}

impl AppContext {

    pub fn new(config: Config) -> Self {
        let index_manager = Arc::new(init_index_manager());
        let node_registry = Arc::new(init_node_registry(index_manager.clone()));
        Self {
            config,
            node_registry,
            index_manager,
        }
    }
}

