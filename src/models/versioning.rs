use crate::macros::key;
use chrono::{DateTime, Utc};
use lmdb::{Cursor, Database, Environment, Transaction, WriteFlags};
use serde::Serialize;
use std::ops::Deref;
use std::sync::Arc;

use utoipa::ToSchema;

use super::collection_transaction::ExplicitTransactionID;
use super::tree_map::TreeMapKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, ToSchema)]
#[schema(value_type = u32, description = "Version number")]
pub struct VersionNumber(u32);

impl From<u32> for VersionNumber {
    fn from(inner: u32) -> Self {
        Self(inner)
    }
}

impl Deref for VersionNumber {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

impl TreeMapKey for VersionNumber {
    fn key(&self) -> u64 {
        self.0 as u64
    }
}

#[derive(Debug, Clone)]
pub enum VersionSource {
    /// Created by an explicit transaction
    Explicit {
        transaction_id: ExplicitTransactionID,
    },
    /// Created by implicit transaction epoch
    Implicit { epoch_id: u32 },
}

#[derive(Debug, Clone)]
pub struct VersionInfo {
    pub version: VersionNumber,
    pub source: VersionSource,

    // Timing information
    pub created_at: DateTime<Utc>,

    // Operation statistics
    pub records_upserted: u32,
    pub records_deleted: u32,
    pub total_operations: u32,
}

impl VersionInfo {
    pub fn new_implicit(
        version: VersionNumber,
        epoch_id: u32,
        created_at: DateTime<Utc>,
        records_upserted: u32,
        records_deleted: u32,
        total_operations: u32,
    ) -> Self {
        Self {
            version,
            source: VersionSource::Implicit { epoch_id },
            created_at,
            records_upserted,
            records_deleted,
            total_operations,
        }
    }

    pub fn new_explicit(
        version: VersionNumber,
        transaction_id: ExplicitTransactionID,
        created_at: DateTime<Utc>,
        records_upserted: u32,
        records_deleted: u32,
        total_operations: u32,
    ) -> Self {
        Self {
            version,
            source: VersionSource::Explicit { transaction_id },
            created_at,
            records_upserted,
            records_deleted,
            total_operations,
        }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(29);

        result.extend_from_slice(&self.version.to_le_bytes());
        match self.source {
            VersionSource::Explicit { transaction_id } => {
                result.push(0);
                result.extend_from_slice(&transaction_id.to_le_bytes());
            }
            VersionSource::Implicit { epoch_id } => {
                result.push(1);
                result.extend_from_slice(&epoch_id.to_le_bytes());
            }
        }

        result.extend_from_slice(&self.created_at.timestamp().to_le_bytes());

        result.extend_from_slice(&self.records_upserted.to_le_bytes());
        result.extend_from_slice(&self.records_deleted.to_le_bytes());
        result.extend_from_slice(&self.total_operations.to_le_bytes());

        result
    }

    fn deserialize(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != 29 {
            return Err("Input must be exactly 29 bytes");
        }

        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let explicit = bytes[4] == 0;

        let source = if explicit {
            VersionSource::Explicit {
                transaction_id: ExplicitTransactionID::from(u32::from_le_bytes(
                    bytes[5..9].try_into().unwrap(),
                )),
            }
        } else {
            VersionSource::Implicit {
                epoch_id: u32::from_le_bytes(bytes[5..9].try_into().unwrap()),
            }
        };

        let created_at_timestamp = i64::from_le_bytes(bytes[9..17].try_into().unwrap());
        let created_at =
            DateTime::from_timestamp(created_at_timestamp, 0).ok_or("Invalid Timestamp")?;

        let records_upserted = u32::from_le_bytes(bytes[17..21].try_into().unwrap());
        let records_deleted = u32::from_le_bytes(bytes[21..25].try_into().unwrap());
        let total_operations = u32::from_le_bytes(bytes[25..29].try_into().unwrap());

        Ok(VersionInfo {
            version: VersionNumber(version),
            source,
            created_at,
            records_upserted,
            records_deleted,
            total_operations,
        })
    }
}

pub struct VersionControl {
    pub env: Arc<Environment>,
    pub db: Database,
}

impl VersionControl {
    pub fn new(env: Arc<Environment>, db: Database) -> lmdb::Result<(Self, VersionNumber)> {
        let version = VersionNumber(0);
        let version_meta = VersionInfo::new_implicit(version, u32::MAX, Utc::now(), 0, 0, 0);

        let version_key = key!(v:version);
        let current_version_key = key!(m:current_version);
        let version_bytes = version_meta.serialize();

        let mut txn = env.begin_rw_txn()?;
        txn.put(db, &version_key, &version_bytes, WriteFlags::empty())?;
        txn.put(
            db,
            &current_version_key,
            &version.to_le_bytes(),
            WriteFlags::empty(),
        )?;
        txn.commit()?;

        Ok((Self { env, db }, version))
    }

    pub fn from_existing(env: Arc<Environment>, db: Database) -> Self {
        Self { env, db }
    }

    fn get_current_version_inner(
        &self,
        txn: &impl lmdb::Transaction,
    ) -> lmdb::Result<VersionNumber> {
        let current_version_key = key!(m:current_version);

        let bytes = match txn.get(self.db, &current_version_key) {
            Ok(bytes) => bytes.try_into().map_err(|_| lmdb::Error::Invalid)?,
            Err(err) => return Err(err),
        };

        let version = VersionNumber(u32::from_le_bytes(bytes));
        Ok(version)
    }

    pub fn get_current_version(&self) -> lmdb::Result<VersionNumber> {
        let txn = self.env.begin_ro_txn()?;
        let version = self.get_current_version_inner(&txn)?;
        txn.abort();
        Ok(version)
    }

    pub fn set_current_version_explicit(
        &self,
        version: VersionNumber,
        transaction_id: ExplicitTransactionID,
        records_upserted: u32,
        records_deleted: u32,
        total_operations: u32,
    ) -> lmdb::Result<()> {
        let mut txn = self.env.begin_rw_txn()?;
        let current_version_key = key!(m:current_version);
        let version_key = key!(v:version);

        let version_info = VersionInfo::new_explicit(
            version,
            transaction_id,
            Utc::now(),
            records_upserted,
            records_deleted,
            total_operations,
        );
        let version_info_serialized = version_info.serialize();

        txn.put(
            self.db,
            &current_version_key,
            &version.to_le_bytes(),
            WriteFlags::empty(),
        )?;
        txn.put(
            self.db,
            &version_key,
            &version_info_serialized,
            WriteFlags::empty(),
        )?;

        txn.commit()?;
        Ok(())
    }

    pub fn set_current_version_implicit(
        &self,
        version: VersionNumber,
        epoch_id: u32,
    ) -> lmdb::Result<()> {
        let mut txn = self.env.begin_rw_txn()?;
        let current_version_key = key!(m:current_version);
        let version_key = key!(v:version);

        let version_info = VersionInfo::new_implicit(version, epoch_id, Utc::now(), 0, 0, 0);
        let version_info_serialized = version_info.serialize();

        txn.put(
            self.db,
            &current_version_key,
            &version.to_le_bytes(),
            WriteFlags::empty(),
        )?;
        txn.put(
            self.db,
            &version_key,
            &version_info_serialized,
            WriteFlags::empty(),
        )?;

        txn.commit()?;
        Ok(())
    }

    pub fn update_version_metadata(
        &self,
        version: VersionNumber,
        records_upserted: u32,
        records_deleted: u32,
        total_operations: u32,
    ) -> lmdb::Result<()> {
        let mut txn = self.env.begin_rw_txn()?;
        let version_key = key!(v:version);

        let bytes = txn.get(self.db, &version_key)?;
        let mut version_info = VersionInfo::deserialize(bytes).unwrap();
        version_info.records_upserted = records_upserted;
        version_info.records_deleted = records_deleted;
        version_info.total_operations = total_operations;
        txn.put(
            self.db,
            &version_key,
            &version_info.serialize(),
            WriteFlags::empty(),
        )?;

        txn.commit()?;
        Ok(())
    }

    pub fn get_version(&self, version: VersionNumber) -> lmdb::Result<VersionInfo> {
        let txn = self.env.begin_ro_txn()?;
        let version_key = key!(v:version);
        let bytes = txn.get(self.db, &version_key)?;
        let info = VersionInfo::deserialize(bytes).unwrap();
        Ok(info)
    }

    pub fn get_versions(&self) -> lmdb::Result<Vec<VersionInfo>> {
        let txn = self.env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(self.db)?;
        let current_version = self.get_current_version_inner(&txn)?;
        let mut versions = Vec::with_capacity(*current_version as usize);
        for (k, v) in cursor.iter_from(&key!(v:VersionNumber(0))) {
            if k.len() != 5 {
                break;
            }

            if k[0] != 0 {
                break;
            }

            versions.push(VersionInfo::deserialize(v).map_err(|_| lmdb::Error::Corrupted)?);
        }
        versions.sort_unstable_by_key(|v| *v.version);
        Ok(versions)
    }

    pub fn get_versions_starting_from_exclusive(
        &self,
        from_version: VersionNumber,
    ) -> lmdb::Result<Vec<VersionInfo>> {
        Ok(self
            .get_versions()?
            .into_iter()
            .skip_while(|v| v.version != from_version) // Skip until the starting version
            .skip(1) // Skip the starting version itself
            .collect())
    }
}
