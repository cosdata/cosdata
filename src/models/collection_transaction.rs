use std::{fmt, fs, mem, ops::Deref, sync::mpsc, thread};

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
    meta_persist::{update_background_version, update_current_version},
    tree_map::TreeMapKey,
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
        collection.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExplicitTransactionID(u32);

impl TreeMapKey for ExplicitTransactionID {
    fn key(&self) -> u64 {
        self.0 as u64
    }
}

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
        collection.transaction_status_map.insert(
            *collection.current_version.read(),
            &id,
            RwLock::new(TransactionStatus::NotStarted {
                last_updated: Utc::now(),
            }),
        );
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

pub struct ImplicitTransactionData {
    version: VersionNumber,
    thread_handle: thread::JoinHandle<Result<DurableWALFile, WaCustomError>>,
    channel: mpsc::Sender<VectorOp>,
}

pub struct ImplicitTransaction {
    data: RwLock<Option<ImplicitTransactionData>>,
}

impl Default for ImplicitTransaction {
    fn default() -> Self {
        Self {
            data: RwLock::new(None),
        }
    }
}

impl ImplicitTransaction {
    pub fn init(&self, collection: &Collection) -> Result<&ImplicitTransactionData, WaCustomError> {
        if let Some(data) = self.data.read().as_ref() {
            return Ok(unsafe {
                mem::transmute::<&ImplicitTransactionData, &ImplicitTransactionData>(data)
            });
        }
        let mut data = self.data.write();
        if let Some(data) = &*data {
            return Ok(unsafe {
                mem::transmute::<&ImplicitTransactionData, &ImplicitTransactionData>(data)
            });
        }
        let mut last_allotted_version = collection.last_allotted_version.write();
        *last_allotted_version = VersionNumber::from(**last_allotted_version + 1);
        let wal = DurableWALFile::new(&collection.get_path(), *last_allotted_version)?;
        wal.flush()?;
        let (tx, rx) = mpsc::channel();
        let version = *last_allotted_version;
        let wal = DurableWALFile::new(&collection.get_path(), version)?;
        let thread_handle = thread::spawn(move || {
            let mut wal = wal;
            for op in rx {
                wal.append(op)?;
            }
            Ok(wal)
        });
        *data = Some(ImplicitTransactionData {
            version,
            thread_handle,
            channel: tx,
        });
        collection
            .vcs
            .set_current_version_implicit(version, random())?;
        *collection.current_version.write() = version;
        update_current_version(&collection.lmdb, version)?;
        Ok(unsafe {
            mem::transmute::<&ImplicitTransactionData, &ImplicitTransactionData>(
                data.as_ref().unwrap(),
            )
        })
    }

    pub fn version(&self, collection: &Collection) -> Result<VersionNumber, WaCustomError> {
        Ok(self.init(collection)?.version)
    }

    pub fn append_to_wal(
        &self,
        collection: &Collection,
        op: VectorOp,
    ) -> Result<(), WaCustomError> {
        let data = self.init(collection)?;
        data.channel.send(op).unwrap();
        Ok(())
    }

    pub fn pre_commit(self, collection: &Collection, config: &Config) -> Result<(), WaCustomError> {
        let Some(data) = self.data.into_inner() else {
            return Ok(());
        };
        if let Some(hnsw_index) = &*collection.hnsw_index.read() {
            hnsw_index.pre_commit_transaction(collection, data.version, config)?;
        }
        if let Some(inverted_index) = &*collection.inverted_index.read() {
            inverted_index.pre_commit_transaction(collection, data.version, config)?;
        }
        if let Some(tf_idf_index) = &*collection.tf_idf_index.read() {
            tf_idf_index.pre_commit_transaction(collection, data.version, config)?;
        }
        collection.flush()?;
        drop(data.channel);
        let wal = data.thread_handle.join().unwrap()?;
        update_background_version(&collection.lmdb, data.version)?;
        collection.vcs.update_version_metadata(
            data.version,
            wal.records_upserted(),
            wal.records_deleted(),
            wal.total_operations(),
        )?;
        drop(wal);
        fs::remove_file(collection.get_path().join(format!("{}.wal", *data.version)))
            .map_err(BufIoError::Io)?;
        Ok(())
    }
}

const NOT_STARTED_MESSAGE: &str = "Indexing has not started yet.";

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ProcessingStats {
    pub records_upserted: u32,
    pub records_deleted: u32,

    pub total_operations: u32,

    // Derived fields
    pub percentage_complete: f32,

    // Timing (optional during progress, required when complete)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_seconds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_throughput: Option<f32>, // records per second
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_processing_rate: Option<f32>, // current rate (for progress)
    #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_completion: Option<DateTime<Utc>>, // only for progress

    // Set when complete
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_created: Option<VersionNumber>,
}

#[derive(Debug, Clone, ToSchema)]
pub enum TransactionStatus {
    NotStarted {
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        last_updated: DateTime<Utc>,
    },
    InProgress {
        stats: ProcessingStats,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        started_at: DateTime<Utc>,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        last_updated: DateTime<Utc>,
    },
    Complete {
        stats: ProcessingStats,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        started_at: DateTime<Utc>,
        #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
        completed_at: DateTime<Utc>,
    },
}

impl TransactionStatus {
    pub fn vector_count(&self) -> u32 {
        match self {
            Self::NotStarted { .. } => 0,
            Self::InProgress { stats, .. } => stats.records_upserted,
            Self::Complete { stats, .. } => stats.records_upserted,
        }
    }

    pub fn increment_vector_count(&mut self, count: u32) {
        match self {
            Self::NotStarted { .. } => {}
            Self::InProgress { stats, .. } => stats.records_upserted += count,
            Self::Complete { stats, .. } => stats.records_upserted += count,
        }
    }

    pub fn update_last_updated(&mut self) {
        match self {
            Self::NotStarted { last_updated } => *last_updated = Utc::now(),
            Self::InProgress { last_updated, .. } => *last_updated = Utc::now(),
            _ => {}
        }
    }

    pub fn complete(&mut self, version_created: VersionNumber) {
        match self {
            Self::NotStarted { .. } => {
                *self = Self::Complete {
                    stats: ProcessingStats {
                        total_operations: 0,
                        records_upserted: 0,
                        records_deleted: 0,
                        percentage_complete: 100.0,
                        processing_time_seconds: Some(0),
                        average_throughput: Some(1.0),
                        current_processing_rate: None,
                        estimated_completion: None,
                        version_created: Some(version_created),
                    },
                    started_at: Utc::now(),
                    completed_at: Utc::now(),
                }
            }
            Self::InProgress {
                stats, started_at, ..
            } => {
                let completed_at = Utc::now();
                let processing_time_seconds = (completed_at - *started_at).num_seconds() as u32;
                *self = Self::Complete {
                    stats: ProcessingStats {
                        version_created: Some(version_created),
                        processing_time_seconds: Some(processing_time_seconds),
                        average_throughput: Some(
                            stats.records_upserted as f32 / processing_time_seconds as f32,
                        ),
                        current_processing_rate: None,
                        estimated_completion: None,
                        ..stats.clone()
                    },
                    started_at: *started_at,
                    completed_at,
                }
            }
            Self::Complete { stats, .. } => {
                stats.version_created = Some(version_created);
            }
        }
    }
}

impl Serialize for TransactionStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::NotStarted { last_updated } => {
                let mut s = serializer.serialize_struct("TransactionStatus", 3)?;
                s.serialize_field("status", "not_started")?;
                s.serialize_field("message", NOT_STARTED_MESSAGE)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
            Self::InProgress {
                started_at,
                stats,
                last_updated,
            } => {
                let mut s = serializer.serialize_struct("TransactionStatus", 3)?;
                s.serialize_field("status", "indexing_in_progress")?;
                s.serialize_field("stats", stats)?;
                s.serialize_field("started_at", started_at)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
            Self::Complete {
                started_at,
                stats,
                completed_at: last_updated,
            } => {
                let mut s = serializer.serialize_struct("TransactionStatus", 3)?;
                s.serialize_field("status", "complete")?;
                s.serialize_field("stats", stats)?;
                s.serialize_field("started_at", started_at)?;
                s.serialize_field("last_updated", last_updated)?;
                s.end()
            }
        }
    }
}
