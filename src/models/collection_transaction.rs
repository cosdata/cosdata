use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use chrono::{DateTime, Utc};
use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};

use crate::{config_loader::Config, indexes::IndexOps};

use super::{
    collection::Collection,
    common::{TSHashTable, WaCustomError},
    prob_node::{ProbNode, SharedNode},
    types::InternalId,
    versioning::{Hash, Version},
    wal::WALFile,
};

pub struct BackgroundCollectionTransaction {
    pub id: Hash,
    pub version_number: u16,
    pub lazy_item_versions_table: Arc<TSHashTable<(InternalId, u16, u8), SharedNode>>,
    level_0_node_offset_counter: AtomicU32,
    node_offset_counter: AtomicU32,
    node_size: u32,
    level_0_node_size: u32,
}

impl BackgroundCollectionTransaction {
    pub fn new(collection: Arc<Collection>) -> Result<Self, WaCustomError> {
        let branch_info = collection.vcs.get_branch_info("main")?.unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = collection
            .vcs
            .generate_hash("main", Version::from(version_number))?;

        let level_0_node_offset_counter = AtomicU32::new(0);
        let node_offset_counter = AtomicU32::new(0);

        let (node_size, level_0_node_size) =
            if let Some(hnsw_index) = &*collection.hnsw_index.read() {
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

    pub fn from_version_id_and_number(
        collection: Arc<Collection>,
        version_id: Hash,
        version_number: u16,
    ) -> Result<Self, WaCustomError> {
        let level_0_node_offset_counter = AtomicU32::new(0);
        let node_offset_counter = AtomicU32::new(0);

        let (node_size, level_0_node_size) =
            if let Some(hnsw_index) = &*collection.hnsw_index.read() {
                let hnsw_params = hnsw_index.hnsw_params.read().unwrap();
                (
                    ProbNode::get_serialized_size(hnsw_params.neighbors_count) as u32,
                    ProbNode::get_serialized_size(hnsw_params.level_0_neighbors_count) as u32,
                )
            } else {
                (0, 0)
            };

        Ok(Self {
            id: version_id,
            version_number,
            level_0_node_offset_counter,
            node_offset_counter,
            node_size,
            level_0_node_size,
            lazy_item_versions_table: Arc::new(TSHashTable::new(16)),
        })
    }

    pub fn pre_commit(self, collection: &Collection, config: &Config) -> Result<(), WaCustomError> {
        if let Some(hnsw_index) = &*collection.hnsw_index.read() {
            hnsw_index.pre_commit_transaction(collection, &self, config)?;
        }
        if let Some(inverted_index) = &*collection.inverted_index.read() {
            inverted_index.pre_commit_transaction(collection, &self, config)?;
        }
        if let Some(tf_idf_index) = &*collection.tf_idf_index.read() {
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

pub struct CollectionTransaction {
    pub id: Hash,
    pub version_number: u16,
    pub wal: WALFile,
}

impl CollectionTransaction {
    pub fn new(collection: Arc<Collection>) -> Result<Self, WaCustomError> {
        let branch_info = collection.vcs.get_branch_info("main")?.unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = collection
            .vcs
            .generate_hash("main", Version::from(version_number))?;

        Ok(Self {
            id,
            version_number,
            wal: WALFile::new(&collection.get_path(), id)?,
        })
    }

    pub fn pre_commit(self) -> Result<(), WaCustomError> {
        self.wal.flush()?;
        Ok(())
    }
}

const NOT_STARTED_MESSAGE: &str = "Indexing has not started yet.";

#[derive(Debug, Clone)]
pub enum TransactionStatus {
    NotStarted {
        last_updated: DateTime<Utc>,
    },
    InProgress {
        progress: Progress,
        last_updated: DateTime<Utc>,
    },
    Complete {
        summary: Summary,
        last_updated: DateTime<Utc>,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Progress {
    pub percentage_done: f32,
    pub records_indexed: u32,
    pub total_records: u32,
    pub rate_per_second: f32,
    pub estimated_time_remaining_seconds: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Summary {
    pub total_records_indexed: u32,
    pub duration_seconds: u32,
    pub average_rate_per_second: f32,
}

impl Serialize for TransactionStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::NotStarted { last_updated } => {
                let mut s = serializer.serialize_struct("IndexStatus", 3)?;
                s.serialize_field("status", "not_started")?;
                s.serialize_field("message", NOT_STARTED_MESSAGE)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
            Self::InProgress {
                progress,
                last_updated,
            } => {
                let mut s = serializer.serialize_struct("IndexStatus", 3)?;
                s.serialize_field("status", "indexing_in_progress")?;
                s.serialize_field("progress", progress)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
            Self::Complete {
                summary,
                last_updated,
            } => {
                let mut s = serializer.serialize_struct("IndexStatus", 3)?;
                s.serialize_field("status", "complete")?;
                s.serialize_field("summary", summary)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
        }
    }
}
