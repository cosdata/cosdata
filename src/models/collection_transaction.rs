use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};

use crate::{config_loader::Config, indexes::IndexOps};

use super::{
    collection::Collection,
    common::{TSHashTable, WaCustomError},
    prob_node::SharedNode,
    types::InternalId,
    versioning::{VersionHash, VersionNumber},
    wal::WALFile,
};

pub struct BackgroundCollectionTransaction {
    pub id: VersionHash,
    pub version_number: VersionNumber,
    pub lazy_item_versions_table: Arc<TSHashTable<(InternalId, VersionHash, u8), SharedNode>>,
}

impl BackgroundCollectionTransaction {
    pub fn new(collection: Arc<Collection>) -> Result<Self, WaCustomError> {
        let branch_info = collection.vcs.get_branch_info("main")?.unwrap();
        let version_number = VersionNumber::from(*branch_info.get_current_version() + 1);
        let id = collection.vcs.generate_hash("main", version_number)?;

        Ok(Self::from_version_id_and_number(
            collection,
            id,
            version_number,
        ))
    }

    pub fn from_version_id_and_number(
        collection: Arc<Collection>,
        version_hash: VersionHash,
        version_number: VersionNumber,
    ) -> Self {
        if let Some(hnsw_index) = &*collection.hnsw_index.read() {
            hnsw_index.offset_counter.write().unwrap().next_file_id();
        }

        Self {
            id: version_hash,
            version_number,
            lazy_item_versions_table: Arc::new(TSHashTable::new(16)),
        }
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
}

pub struct CollectionTransaction {
    pub id: VersionHash,
    pub version_number: VersionNumber,
    pub wal: WALFile,
}

impl CollectionTransaction {
    pub fn new(collection: Arc<Collection>) -> Result<Self, WaCustomError> {
        let branch_info = collection.vcs.get_branch_info("main")?.unwrap();
        let version_number = VersionNumber::from(*branch_info.get_current_version() + 1);
        let id = collection.vcs.generate_hash("main", version_number)?;

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
