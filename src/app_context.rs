use std::path::Path;
use std::sync::Arc;

use crate::config_loader::Config;
use crate::models::buffered_io::BufferManagerFactory;
use crate::models::cache_loader::NodeRegistry;

fn init_node_registry() -> NodeRegistry {
    let bufmans = Arc::new(BufferManagerFactory::new(
        Path::new(".").into(),
        |root, ver| root.join(format!("{}.index", **ver)),
    ));
    NodeRegistry::new(1000, bufmans)
}

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub node_registry: Arc<NodeRegistry>,
}

impl AppContext {

    pub fn new(config: Config) -> Self {
        Self {
            config,
            node_registry: Arc::new(init_node_registry()),
        }
    }
}

