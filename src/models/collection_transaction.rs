use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use crate::{config_loader::Config, indexes::IndexOps};

use super::{
    collection::Collection,
    common::{TSHashTable, WaCustomError},
    prob_node::{ProbNode, SharedNode},
    types::InternalId,
    versioning::{Hash, Version},
};

pub struct CollectionTransaction {
    pub id: Hash,
    pub version_number: u16,
    pub lazy_item_versions_table: Arc<TSHashTable<(InternalId, u16, u8), SharedNode>>,
    level_0_node_offset_counter: AtomicU32,
    node_offset_counter: AtomicU32,
    node_size: u32,
    level_0_node_size: u32,
}

impl CollectionTransaction {
    pub fn new(collection: Arc<Collection>) -> Result<Self, WaCustomError> {
        let branch_info = collection.vcs.get_branch_info("main")?.unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = collection
            .vcs
            .generate_hash("main", Version::from(version_number))?;

        let level_0_node_offset_counter = AtomicU32::new(0);
        let node_offset_counter = AtomicU32::new(0);

        let (node_size, level_0_node_size) =
            if let Some(hnsw_index) = &*collection.hnsw_index.read().unwrap() {
                let hnsw_params = hnsw_index.hnsw_params.read().unwrap();
                (
                    ProbNode::get_serialized_size(hnsw_params.neighbors_count) as u32,
                    ProbNode::get_serialized_size(hnsw_params.level_0_neighbors_count) as u32,
                )
            } else {
                (0, 0)
            };

        Ok(Self {
            id,
            version_number,
            level_0_node_offset_counter,
            node_offset_counter,
            node_size,
            level_0_node_size,
            lazy_item_versions_table: Arc::new(TSHashTable::new(16)),
        })
    }

    pub fn pre_commit(self, collection: &Collection, config: &Config) -> Result<(), WaCustomError> {
        if let Some(hnsw_index) = &*collection.hnsw_index.read().unwrap() {
            hnsw_index.pre_commit_transaction(collection, &self, config)?;
        }
        if let Some(inverted_index) = &*collection.inverted_index.read().unwrap() {
            inverted_index.pre_commit_transaction(collection, &self, config)?;
        }
        if let Some(tf_idf_index) = &*collection.tf_idf_index.read().unwrap() {
            tf_idf_index.pre_commit_transaction(collection, &self, config)?;
        }
        collection.flush(config)?;
        Ok(())
    }

    pub fn get_new_node_offset(&self) -> u32 {
        self.node_offset_counter
            .fetch_add(self.node_size, Ordering::Relaxed)
    }

    pub fn get_new_level_0_node_offset(&self) -> u32 {
        self.level_0_node_offset_counter
            .fetch_add(self.level_0_node_size, Ordering::Relaxed)
    }
}
