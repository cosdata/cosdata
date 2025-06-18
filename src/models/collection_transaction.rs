use std::{fmt, fs, mem, ops::Deref};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rand::random;
use serde::{
    de::{self, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use utoipa::{
    openapi::{ObjectBuilder, RefOr, Schema, Type},
    PartialSchema, ToSchema,
};

use crate::{config_loader::Config, indexes::IndexOps};

use super::{
    buffered_io::BufIoError,
    collection::Collection,
    common::WaCustomError,
    durable_wal::DurableWALFile,
    versioning::VersionNumber,
    wal::{VectorOp, WALFile},
};

pub struct BackgroundExplicitTransaction {
    pub version: VersionNumber,
}

impl BackgroundExplicitTransaction {
    pub fn from_version_id_and_number(collection: &Collection, version: VersionNumber) -> Self {
        if let Some(hnsw_index) = &*collection.hnsw_index.read() {
            hnsw_index.offset_counter.write().unwrap().next_file_id();
        }

        Self { version }
    }

    pub fn pre_commit(self, collection: &Collection, config: &Config) -> Result<(), WaCustomError> {
        if let Some(hnsw_index) = &*collection.hnsw_index.read() {
            hnsw_index.pre_commit_transaction(collection, self.version, config)?;
        }
        if let Some(inverted_index) = &*collection.inverted_index.read() {
            inverted_index.pre_commit_transaction(collection, self.version, config)?;
        }
        if let Some(tf_idf_index) = &*collection.tf_idf_index.read() {
            tf_idf_index.pre_commit_transaction(collection, self.version, config)?;
        }
        collection.flush(config)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExplicitTransactionID(u32);

impl ToSchema for ExplicitTransactionID {
    fn name() -> std::borrow::Cow<'static, str> {
        "Transaction ID".into()
    }
}

impl PartialSchema for ExplicitTransactionID {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::Object(
            ObjectBuilder::new().schema_type(Type::String).build(),
        ))
    }
}

// Serialize to `"0xDEADBEEF"`
impl Serialize for ExplicitTransactionID {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex_str = format!("0x{:08X}", self.0); // Uppercase with prefix
        serializer.serialize_str(&hex_str)
    }
}

// Deserialize from `"0xDEADBEEF"`
struct ExplicitTransactionIDVisitor;

impl Visitor<'_> for ExplicitTransactionIDVisitor {
    type Value = ExplicitTransactionID;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a hexadecimal string like \"0xDEADBEEF\"")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let v = v
            .strip_prefix("0x")
            .ok_or_else(|| E::custom("missing `0x` prefix"))?;

        if v.len() != 8 {
            return Err(E::custom(
                "hex string must be 8 characters after the `0x` prefix",
            ));
        }

        let val = u32::from_str_radix(v, 16).map_err(E::custom)?;
        Ok(ExplicitTransactionID(val))
    }
}

impl<'de> Deserialize<'de> for ExplicitTransactionID {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ExplicitTransactionIDVisitor)
    }
}

impl Deref for ExplicitTransactionID {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<u32> for ExplicitTransactionID {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

pub struct ExplicitTransaction {
    pub id: ExplicitTransactionID,
    pub wal: WALFile,
}

impl ExplicitTransaction {
    pub fn new(collection: &Collection, config: &Config) -> Result<Self, WaCustomError> {
        let mut current_implicit_txn = collection.current_implicit_transaction.write();
        mem::take(&mut *current_implicit_txn).pre_commit(collection, config)?;
        let id = ExplicitTransactionID(random());
        Ok(Self {
            id,
            wal: WALFile::new()?,
        })
    }

    pub fn pre_commit(
        self,
        collection: &Collection,
        version: VersionNumber,
    ) -> Result<(), WaCustomError> {
        self.wal.flush(&collection.get_path(), version)?;
        Ok(())
    }
}

pub struct ImplicitTransaction {
    version_and_wal: RwLock<Option<(VersionNumber, DurableWALFile)>>,
}

impl Default for ImplicitTransaction {
    fn default() -> Self {
        Self {
            version_and_wal: RwLock::new(None),
        }
    }
}

impl ImplicitTransaction {
    pub fn init(&self, collection: &Collection) -> Result<VersionNumber, BufIoError> {
        if let Some(version_and_wal) = self.version_and_wal.read().as_ref() {
            return Ok(version_and_wal.0);
        }
        let mut version = self.version_and_wal.write();
        let mut last_allotted_version = collection.last_allotted_version.write();
        *last_allotted_version = VersionNumber::from(**last_allotted_version + 1);
        let wal = DurableWALFile::new(&collection.get_path(), *last_allotted_version)?;
        wal.flush()?;
        *version = Some((*last_allotted_version, wal));
        collection.transaction_status_map.insert(
            *last_allotted_version,
            &last_allotted_version,
            RwLock::new(TransactionStatus::InProgress {
                progress: Progress {
                    percentage_done: 0.0,
                    records_indexed: 0,
                    total_records: 0,
                    rate_per_second: 0.0,
                    estimated_time_remaining_seconds: u32::MAX,
                },
                started_at: Utc::now(),
                last_updated: Utc::now(),
            }),
        );
        Ok(*last_allotted_version)
    }

    pub fn append_to_wal(&self, collection: &Collection, op: VectorOp) -> Result<(), BufIoError> {
        self.init(collection)?;
        let version_and_wal = self.version_and_wal.read();
        let version_and_wal = version_and_wal.as_ref().unwrap();
        version_and_wal.1.append(op)?;
        Ok(())
    }

    pub fn pre_commit(self, collection: &Collection, config: &Config) -> Result<(), WaCustomError> {
        let Some((version, wal)) = self.version_and_wal.into_inner() else {
            return Ok(());
        };
        if let Some(hnsw_index) = &*collection.hnsw_index.read() {
            hnsw_index.pre_commit_transaction(collection, version, config)?;
        }
        if let Some(inverted_index) = &*collection.inverted_index.read() {
            inverted_index.pre_commit_transaction(collection, version, config)?;
        }
        if let Some(tf_idf_index) = &*collection.tf_idf_index.read() {
            tf_idf_index.pre_commit_transaction(collection, version, config)?;
        }
        collection.flush(config)?;
        drop(wal);
        fs::remove_file(collection.get_path().join(format!("{}.wal", *version)))
            .map_err(BufIoError::Io)?;
        let status = collection
            .transaction_status_map
            .get_latest(&version)
            .unwrap();
        status.write().complete();
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
        started_at: DateTime<Utc>,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        last_updated: DateTime<Utc>,
    },
    Complete {
        summary: Summary,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        started_at: DateTime<Utc>,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        last_updated: DateTime<Utc>,
    },
}

impl TransactionStatus {
    pub fn vector_count(&self) -> u32 {
        match self {
            Self::NotStarted { .. } => 0,
            Self::InProgress { progress, .. } => progress.records_indexed,
            Self::Complete { summary, .. } => summary.total_records_indexed,
        }
    }

    pub fn increment_vector_count(&mut self, count: u32) {
        match self {
            Self::NotStarted { .. } => {}
            Self::InProgress { progress, .. } => progress.records_indexed += count,
            Self::Complete { summary, .. } => summary.total_records_indexed += count,
        }
    }

    pub fn update_last_updated(&mut self) {
        match self {
            Self::NotStarted { last_updated } => *last_updated = Utc::now(),
            Self::InProgress { last_updated, .. } => *last_updated = Utc::now(),
            Self::Complete { last_updated, .. } => *last_updated = Utc::now(),
        }
    }

    pub fn complete(&mut self) {
        match self {
            Self::NotStarted { .. } => {
                *self = Self::Complete {
                    summary: Summary {
                        total_records_indexed: 0,
                        duration_seconds: 0,
                        average_rate_per_second: 0.0,
                    },
                    started_at: Utc::now(),
                    last_updated: Utc::now(),
                }
            }
            Self::InProgress {
                progress,
                started_at,
                ..
            } => {
                *self = Self::Complete {
                    summary: Summary {
                        total_records_indexed: progress.records_indexed,
                        duration_seconds: (Utc::now() - *started_at).num_seconds() as u32,
                        average_rate_per_second: progress.records_indexed as f32
                            / ((Utc::now() - *started_at).num_seconds() as u32).max(1) as f32,
                    },
                    started_at: *started_at,
                    last_updated: Utc::now(),
                }
            }
            Self::Complete { .. } => {}
        }
    }
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
                started_at,
                progress,
                last_updated,
            } => {
                let mut s = serializer.serialize_struct("IndexStatus", 3)?;
                s.serialize_field("status", "indexing_in_progress")?;
                s.serialize_field("progress", progress)?;
                s.serialize_field("started_at", started_at)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
            Self::Complete {
                started_at,
                summary,
                last_updated,
            } => {
                let mut s = serializer.serialize_struct("IndexStatus", 3)?;
                s.serialize_field("status", "complete")?;
                s.serialize_field("summary", summary)?;
                s.serialize_field("started_at", started_at)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
        }
    }
}
