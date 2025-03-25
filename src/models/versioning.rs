use crate::macros::key;
use lmdb::{Cursor, Database, Environment, RoTransaction, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher24;
use std::hash::Hasher;
use std::ops::Deref;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BranchId(u64);

impl From<u64> for BranchId {
    fn from(inner: u64) -> Self {
        Self(inner)
    }
}

impl Deref for BranchId {
    type Target = u64;

    fn deref(&self) -> &u64 {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version(u32);

impl From<u32> for Version {
    fn from(inner: u32) -> Self {
        Self(inner)
    }
}

impl Deref for Version {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestamp(pub u32);

impl From<u32> for Timestamp {
    fn from(inner: u32) -> Self {
        Self(inner)
    }
}

impl Deref for Timestamp {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hash(u32);

impl From<u32> for Hash {
    fn from(inner: u32) -> Self {
        Self(inner)
    }
}

impl Deref for Hash {
    type Target = u32;

    fn deref(&self) -> &u32 {
        &self.0
    }
}

impl BranchId {
    pub fn new(branch_name: &str) -> Self {
        let mut hasher = SipHasher24::new();
        hasher.write(branch_name.as_bytes());
        BranchId(hasher.finish())
    }
}

impl Timestamp {
    fn now() -> Self {
        Timestamp(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as u32,
        )
    }
}

#[derive(Debug, Clone)]
pub struct VersionHash {
    pub branch: BranchId,
    pub version: Version,
    pub timestamp: Timestamp,
}

impl VersionHash {
    pub fn new(branch: BranchId, version: Version) -> Self {
        Self {
            branch,
            version,
            timestamp: Timestamp::now(),
        }
    }

    pub fn calculate_hash(&self) -> Hash {
        let mut hasher = SipHasher24::new();
        hasher.write(&self.branch.to_le_bytes());
        hasher.write(&self.version.to_le_bytes());
        hasher.write(&self.timestamp.to_le_bytes());
        let hash = (hasher.finish() & 0xFFFFFFFF) as u32;
        Hash(hash)
    }

    fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(16);

        result.extend_from_slice(&self.branch.to_le_bytes());
        result.extend_from_slice(&self.version.to_le_bytes());
        result.extend_from_slice(&self.timestamp.to_le_bytes());

        result
    }

    fn deserialize(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != 16 {
            return Err("Input must be exactly 16 bytes");
        }

        let branch = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let timestamp = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

        Ok(VersionHash {
            branch: BranchId(branch),
            version: Version(version),
            timestamp: Timestamp(timestamp),
        })
    }
}

#[derive(Debug, Clone)]
pub struct BranchInfo {
    pub branch_name: String,
    pub current_version: Version,
    pub parent_branch: BranchId,
    pub parent_version: Version,
    // The least version, not the first or root version, because all the keys in LMDB are ordered,
    // when iterating through all the versions we start from this version.
    pub least_version: Hash,
}

impl BranchInfo {
    fn serialize(&self) -> Vec<u8> {
        let name_bytes = self.branch_name.as_bytes();

        let mut result = Vec::with_capacity(20 + name_bytes.len());

        // Serialize current_version
        result.extend_from_slice(&self.current_version.to_le_bytes());

        // Serialize parent_branch
        result.extend_from_slice(&self.parent_branch.to_le_bytes());

        // Serialize parent_version
        result.extend_from_slice(&self.parent_version.to_le_bytes());

        // Serialize least_version
        result.extend_from_slice(&self.least_version.to_le_bytes());

        // Serialize branch_name
        result.extend_from_slice(name_bytes);

        result
    }

    fn deserialize(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 20 {
            return Err("Input must be at least 20 bytes");
        }

        // Deserialize current_version
        let current_version = Version(u32::from_le_bytes(bytes[0..4].try_into().unwrap()));

        // Deserialize parent_branch
        let parent_branch = BranchId(u64::from_le_bytes(bytes[4..12].try_into().unwrap()));

        // Deserialize parent_version
        let parent_version = Version(u32::from_le_bytes(bytes[12..16].try_into().unwrap()));

        // Deserialize least_version
        let least_version = Hash(u32::from_le_bytes(bytes[16..20].try_into().unwrap()));

        let branch_name =
            String::from_utf8(bytes[20..].to_vec()).map_err(|_| "Invalid UTF-8 in branch name")?;

        Ok(BranchInfo {
            branch_name,
            current_version,
            parent_branch,
            parent_version,
            least_version,
        })
    }

    pub fn get_current_version(&self) -> Version {
        self.current_version
    }
}

pub struct VersionControl {
    pub env: Arc<Environment>,
    pub db: Arc<Database>,
}

#[allow(unused)]
impl VersionControl {
    pub fn new(env: Arc<Environment>, db: Arc<Database>) -> lmdb::Result<(Self, Hash)> {
        let main_branch_id = BranchId::new("main");
        let current_version = Version(0);
        let current_version_hash = VersionHash::new(main_branch_id, current_version);
        let hash = current_version_hash.calculate_hash();
        let branch_info = BranchInfo {
            branch_name: "main".to_string(),
            current_version,
            parent_branch: main_branch_id,
            parent_version: Version(0),
            least_version: hash,
        };

        let branch_key = key!(b:main_branch_id);
        let branch_bytes = branch_info.serialize();

        let version_key = key!(v:hash);
        let version_bytes = current_version_hash.serialize();

        let mut txn = env.begin_rw_txn()?;
        txn.put(*db, &branch_key, &branch_bytes, WriteFlags::empty())?;
        txn.put(*db, &version_key, &version_bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok((Self { env, db }, hash))
    }

    pub fn from_existing(env: Arc<Environment>, db: Arc<Database>) -> Self {
        Self { env, db }
    }

    pub fn generate_hash(&self, branch_name: &str, version: Version) -> lmdb::Result<Hash> {
        let branch_id = BranchId::new(branch_name);
        let version_hash = VersionHash::new(branch_id, version);
        let hash = version_hash.calculate_hash();
        let version_key = key!(v:hash);
        let bytes = version_hash.serialize();

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(*self.db, &version_key, &bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok(hash)
    }

    pub fn add_next_version(&self, branch_name: &str) -> lmdb::Result<(Hash, Version)> {
        let branch_id = BranchId::new(branch_name);
        let branch_key = key!(b:branch_id);

        let mut txn = self.env.begin_rw_txn()?;
        let bytes = txn.get(*self.db, &branch_key)?;

        let mut branch_info: BranchInfo = BranchInfo::deserialize(bytes).unwrap();
        let new_version = Version(*branch_info.current_version + 1);
        let version_hash = VersionHash::new(branch_id, new_version);
        let hash = version_hash.calculate_hash();
        branch_info.current_version = new_version;
        if hash.to_le_bytes() < branch_info.least_version.to_le_bytes() {
            branch_info.least_version = hash;
        }
        let bytes = branch_info.serialize();

        txn.put(*self.db, &branch_key, &bytes, WriteFlags::empty())?;

        let version_key = key!(v:hash);
        let bytes = version_hash.serialize();

        txn.put(*self.db, &version_key, &bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok((hash, new_version))
    }

    pub fn create_new_branch(
        &mut self,
        branch_name: &str,
        parent_branch_name: &str,
    ) -> lmdb::Result<()> {
        let branch_id = BranchId::new(branch_name);
        let parent_branch_id = BranchId::new(parent_branch_name);
        let branch_key = key!(b:branch_id);
        let parent_key = key!(b:parent_branch_id);

        let mut txn = self.env.begin_rw_txn()?;

        if txn.get(*self.db, &branch_key) != Err(lmdb::Error::NotFound) {
            return Err(lmdb::Error::KeyExist);
        }

        let parent_bytes = txn.get(*self.db, &parent_key)?;
        let parent_info = BranchInfo::deserialize(parent_bytes).unwrap();
        let current_version = Version(0);
        let current_version_hash = VersionHash::new(branch_id, current_version);
        let hash = current_version_hash.calculate_hash();
        let new_branch_info = BranchInfo {
            branch_name: branch_name.to_string(),
            current_version: Version(0),
            parent_branch: parent_branch_id,
            parent_version: parent_info.current_version,
            least_version: hash,
        };

        let bytes = new_branch_info.serialize();

        let version_key = key!(v:hash);
        let version_bytes = current_version_hash.serialize();

        txn.put(*self.db, &branch_key, &bytes, WriteFlags::empty())?;
        txn.put(*self.db, &version_key, &version_bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok(())
    }

    pub fn branch_exists(&self, branch_name: &str) -> lmdb::Result<bool> {
        let branch_id = BranchId::new(branch_name);
        let branch_key = key!(b:branch_id);

        let txn = self.env.begin_ro_txn()?;

        let exists = match txn.get(*self.db, &branch_key) {
            Ok(_) => true,
            Err(lmdb::Error::NotFound) => false,
            Err(err) => {
                txn.abort();
                return Err(err);
            }
        };

        txn.abort();

        Ok(exists)
    }

    pub fn get_branch_info(&self, branch_name: &str) -> lmdb::Result<Option<BranchInfo>> {
        let branch_id = BranchId::new(branch_name);
        let branch_key = key!(b:branch_id);

        let txn = self.env.begin_ro_txn()?;

        let bytes = match txn.get(*self.db, &branch_key) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => {
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        let branch_info = BranchInfo::deserialize(bytes).unwrap();

        txn.abort();

        Ok(Some(branch_info))
    }

    pub fn set_branch_version(
        &self,
        branch_name: &str,
        version: Version,
        hash: Hash,
    ) -> lmdb::Result<()> {
        let branch_id = BranchId::new(branch_name);
        let branch_key = key!(b:branch_id);

        let mut txn = self.env.begin_rw_txn()?;

        let bytes = txn.get(*self.db, &branch_key)?;

        let mut branch_info = BranchInfo::deserialize(bytes).unwrap();
        branch_info.current_version = version;
        if hash.to_le_bytes() < branch_info.least_version.to_le_bytes() {
            branch_info.least_version = hash;
        }
        let bytes = branch_info.serialize();

        txn.put(*self.db, &branch_key, &bytes, WriteFlags::empty())?;
        txn.commit()?;
        Ok(())
    }

    pub fn get_version_hash(
        &self,
        hash: &Hash,
        txn: &RoTransaction<'_>,
    ) -> lmdb::Result<Option<VersionHash>> {
        let version_key = key!(v:hash);

        let bytes = match txn.get(*self.db, &version_key) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => {
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        let version_hash = VersionHash::deserialize(bytes).unwrap();

        Ok(Some(version_hash))
    }

    pub fn trace_to_main(&self, start_branch: &str) -> lmdb::Result<Vec<BranchInfo>> {
        let mut branch_path = Vec::new();
        let branch_id = BranchId::new(start_branch);
        let mut current_branch_key = key!(b:branch_id);

        let txn = self.env.begin_ro_txn()?;

        let get_branch_info = |branch_key: Vec<u8>| {
            let bytes = match txn.get(*self.db, &branch_key) {
                Ok(bytes) => bytes,
                Err(lmdb::Error::NotFound) => {
                    return Ok(None);
                }
                Err(err) => {
                    return Err(err);
                }
            };

            let branch_info = BranchInfo::deserialize(bytes).unwrap();

            Ok(Some(branch_info))
        };

        while let Some(info) = get_branch_info(current_branch_key)? {
            if info.branch_name == "main" {
                break;
            }
            current_branch_key = key!(b:info.parent_branch);
            branch_path.push(info);
        }

        branch_path.reverse();
        Ok(branch_path)
    }

    pub fn get_branch_versions(&self, branch_name: &str) -> lmdb::Result<Vec<(Hash, VersionHash)>> {
        let branch_info = self.get_branch_info(branch_name)?.unwrap();
        let txn = self.env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(*self.db)?;
        let mut versions = Vec::with_capacity(*branch_info.current_version as usize);
        for (k, v) in cursor.iter_from(&key!(v:branch_info.least_version)) {
            if k.len() != 5 {
                break;
            }

            if k[0] != 0 {
                break;
            }

            let hash = Hash::from(u32::from_le_bytes(k[1..].try_into().unwrap()));
            let version_hash = VersionHash::deserialize(v).unwrap();
            versions.push((hash, version_hash));
        }
        versions.sort_unstable_by_key(|(_, v)| *v.version);
        Ok(versions)
    }

    pub fn get_branch_versions_starting_from_exclusive(
        &self,
        branch_name: &str,
        from_version: Hash,
    ) -> lmdb::Result<Vec<(Hash, VersionHash)>> {
        Ok(self
            .get_branch_versions(branch_name)?
            .into_iter()
            .skip_while(|(hash, _)| hash != &from_version) // Skip until the starting version
            .skip(1) // Skip the starting version itself
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_version_hashing_function_uniqueness() {
        let branch_id = BranchId::new("main");
        let mut versions = HashSet::new();

        for i in 0..1000 {
            let version_hash = VersionHash::new(branch_id, Version::from(i));
            versions.insert(version_hash.calculate_hash());
            // simulate some processing
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(versions.len(), 1000);
    }
}
