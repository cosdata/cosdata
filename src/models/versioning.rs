use crate::macros::key;
use lmdb::{Cursor, Database, Environment, Transaction, WriteFlags};
use serde::Serialize;
use std::ops::Deref;
use std::sync::Arc;

use super::tree_map::TreeMapKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
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
pub struct VersionInfo {
    pub version: VersionNumber,
    pub implicit: bool,
}

impl VersionInfo {
    pub fn new(version: VersionNumber, implicit: bool) -> Self {
        Self { version, implicit }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(5);

        result.extend_from_slice(&self.version.to_le_bytes());
        result.push(self.implicit as u8);

        result
    }

    fn deserialize(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != 5 {
            return Err("Input must be exactly 5 bytes");
        }

        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let implicit = bytes[4] != 0;

        Ok(VersionInfo {
            version: VersionNumber(version),
            implicit,
        })
    }
}

pub struct VersionControl {
    pub env: Arc<Environment>,
    pub db: Database,
}

#[allow(unused)]
impl VersionControl {
    pub fn new(env: Arc<Environment>, db: Database) -> lmdb::Result<(Self, VersionNumber)> {
        let version = VersionNumber(0);
        let version_meta = VersionInfo::new(version, false);

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

    pub fn set_current_version(&self, version: VersionNumber, implicit: bool) -> lmdb::Result<()> {
        let mut txn = self.env.begin_rw_txn()?;
        let current_version_key = key!(m:current_version);
        let version_key = key!(v:version);

        let version_info = VersionInfo::new(version, implicit);
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

            let hash = VersionNumber::from(u32::from_le_bytes(k[1..].try_into().unwrap()));
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
