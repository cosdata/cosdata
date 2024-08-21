use lmdb::{Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use rkyv::{Archive, Deserialize, Serialize};
use siphasher::sip::SipHasher24;
use std::hash::Hasher;
use std::ops::Deref;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub struct Timestamp(u32);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
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
    fn new(branch_name: &str) -> Self {
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

#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct VersionHash {
    branch: BranchId,
    version: Version,
    timestamp: Timestamp,
}

impl VersionHash {
    fn new(branch: BranchId, version: Version) -> Self {
        Self {
            branch,
            version,
            timestamp: Timestamp::now(),
        }
    }

    fn calculate_hash(&self) -> Hash {
        let branch_last_4_bytes = (*self.branch & 0xFFFFFFFF) as u32;
        Hash(branch_last_4_bytes ^ *self.version ^ *self.timestamp)
    }
}

#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct BranchInfo {
    branch_name: String,
    current_version: Version,
    parent_branch: BranchId,
    parent_version: Version,
}

pub struct VersionControl {
    env: Arc<Environment>,
    // `BranchId` -> `BranchInfo`
    versions_db: Database,
    // `Hash` -> `VersionHash`
    branches_db: Database,
}

impl VersionControl {
    pub fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let main_branch_id = BranchId::new("main");
        let branch_info = BranchInfo {
            branch_name: "main".to_string(),
            current_version: Version(0),
            parent_branch: main_branch_id,
            parent_version: Version(0),
        };

        let key = main_branch_id.to_le_bytes();
        let bytes = rkyv::to_bytes::<_, 256>(&branch_info).unwrap();

        let versions_db = env.create_db(Some("versions"), DatabaseFlags::empty())?;
        let branches_db = env.create_db(Some("branches"), DatabaseFlags::empty())?;

        let mut txn = env.begin_rw_txn()?;
        txn.put(branches_db, &key, &bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok(Self {
            env,
            versions_db,
            branches_db,
        })
    }

    pub fn generate_hash(&self, branch_name: &str, version: Version) -> lmdb::Result<Hash> {
        let branch_id = BranchId::new(branch_name);
        let version_hash = VersionHash::new(branch_id, version);
        let hash = version_hash.calculate_hash();
        let key = hash.to_le_bytes();
        let bytes = rkyv::to_bytes::<_, 256>(&version_hash).unwrap();

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(self.versions_db, &key, &bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok(hash)
    }

    pub fn add_next_version(&self, branch_name: &str) -> lmdb::Result<Hash> {
        let branch_id = BranchId::new(branch_name);
        let key = branch_id.to_le_bytes();

        let mut txn = self.env.begin_rw_txn()?;
        let bytes = txn.get(self.branches_db, &key)?;

        let mut branch_info: BranchInfo = unsafe { rkyv::from_bytes_unchecked(bytes) }.unwrap();
        let new_version = Version(branch_info.current_version + 1);
        branch_info.current_version = new_version;
        let bytes = rkyv::to_bytes::<_, 256>(&branch_info).unwrap();

        txn.put(self.branches_db, &key, &bytes, WriteFlags::empty())?;
        let version_hash = VersionHash::new(branch_id, new_version);
        let hash = version_hash.calculate_hash();

        let key = hash.to_le_bytes();
        let bytes = rkyv::to_bytes::<_, 256>(&version_hash).unwrap();

        txn.put(self.versions_db, &key, &bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok(hash)
    }

    pub fn create_new_branch(
        &mut self,
        branch_name: &str,
        parent_branch_name: &str,
    ) -> lmdb::Result<()> {
        let branch_id = BranchId::new(branch_name);
        let parent_branch_id = BranchId::new(parent_branch_name);
        let key = branch_id.to_le_bytes();
        let parent_key = parent_branch_id.to_le_bytes();

        let mut txn = self.env.begin_rw_txn()?;

        if txn.get(self.branches_db, &key) != Err(lmdb::Error::NotFound) {
            return Err(lmdb::Error::KeyExist);
        }

        let parent_bytes = txn.get(self.branches_db, &parent_key)?;
        let parent_info: BranchInfo = unsafe { rkyv::from_bytes_unchecked(parent_bytes) }.unwrap();

        let new_branch_info = BranchInfo {
            branch_name: branch_name.to_string(),
            current_version: Version(0),
            parent_branch: parent_branch_id,
            parent_version: parent_info.current_version,
        };

        let bytes = rkyv::to_bytes::<_, 256>(&new_branch_info).unwrap();

        txn.put(self.branches_db, &key, &bytes, WriteFlags::empty())?;
        txn.commit()?;

        Ok(())
    }

    pub fn branch_exists(&self, branch_name: &str) -> lmdb::Result<bool> {
        let branch_id = BranchId::new(branch_name);
        let key = branch_id.to_le_bytes();

        let txn = self.env.begin_ro_txn()?;

        let exists = match txn.get(self.branches_db, &key) {
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
        let key = branch_id.to_le_bytes();

        let txn = self.env.begin_ro_txn()?;

        let bytes = match txn.get(self.branches_db, &key) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => {
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        let branch_info = unsafe { rkyv::from_bytes_unchecked(bytes) }.unwrap();

        txn.abort();

        Ok(Some(branch_info))
    }

    pub fn trace_to_main(&self, start_branch: &str) -> lmdb::Result<Vec<BranchInfo>> {
        let mut branch_path = Vec::new();
        let branch_id = BranchId::new(start_branch);
        let mut current_key = branch_id.to_le_bytes();

        let txn = self.env.begin_ro_txn()?;

        let get_branch_info = |key: [u8; 8]| {
            let bytes = match txn.get(self.branches_db, &key) {
                Ok(bytes) => bytes,
                Err(lmdb::Error::NotFound) => {
                    return Ok(None);
                }
                Err(err) => {
                    return Err(err);
                }
            };

            let branch_info: BranchInfo = unsafe { rkyv::from_bytes_unchecked(bytes) }.unwrap();

            Ok(Some(branch_info))
        };

        while let Some(info) = get_branch_info(current_key)? {
            if info.branch_name == "main" {
                break;
            }
            current_key = info.parent_branch.to_le_bytes();
            branch_path.push(info);
        }

        branch_path.reverse();
        Ok(branch_path)
    }
}
