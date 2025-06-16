use chrono::{DateTime, Utc};
use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};
use utoipa::ToSchema;

use crate::{config_loader::Config, indexes::IndexOps};

use super::{
    collection::Collection, common::WaCustomError, meta_persist::retrieve_current_version,
    versioning::VersionNumber, wal::WALFile,
};

pub struct BackgroundExplicitTransaction {
    pub version: VersionNumber,
}

impl BackgroundExplicitTransaction {
    pub fn new(collection: &Collection) -> Result<Self, WaCustomError> {
        let current_version_number = retrieve_current_version(&collection.lmdb)?;
        let version_number = VersionNumber::from(*current_version_number + 1);

        Ok(Self::from_version_id_and_number(collection, version_number))
    }

    pub fn from_version_id_and_number(collection: &Collection, version: VersionNumber) -> Self {
        if let Some(hnsw_index) = &*collection.hnsw_index.read() {
            hnsw_index.offset_counter.write().unwrap().next_file_id();
        }

        Self { version }
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

pub struct ExplicitTransaction {
    pub version: VersionNumber,
    pub wal: WALFile,
}

impl ExplicitTransaction {
    pub fn new(collection: &Collection) -> Result<Self, WaCustomError> {
        let current_version_number = collection.vcs.get_current_version()?;
        let version = VersionNumber::from(*current_version_number + 1);

        Ok(Self {
            version,
            wal: WALFile::new(&collection.get_path(), version)?,
        })
    }

    pub fn pre_commit(self) -> Result<(), WaCustomError> {
        self.wal.flush()?;
        Ok(())
    }
}

const NOT_STARTED_MESSAGE: &str = "Indexing has not started yet.";

#[derive(Debug, Clone, ToSchema)]
pub enum TransactionStatus {
    NotStarted {
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        last_updated: DateTime<Utc>,
    },
    InProgress {
        progress: Progress,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        last_updated: DateTime<Utc>,
    },
    Complete {
        summary: Summary,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        last_updated: DateTime<Utc>,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct Progress {
    pub percentage_done: f32,
    pub records_indexed: u32,
    pub total_records: u32,
    pub rate_per_second: f32,
    pub estimated_time_remaining_seconds: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
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
