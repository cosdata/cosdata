use lmdb::{Environment, Database, DatabaseFlags, WriteFlags, Transaction, Error as LmdbError};
use serde::{Serialize, Deserialize};
use bincode;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BranchInfo {
    branch_name: String,
    current_hash: String,
    parent_branch: String,
    parent_hash: String,
    parent_version: u64,
}

struct VersionHasher {
    env: Environment,
    db: Database,
}

impl VersionHasher {
    fn new(path: &str) -> Result<Self, LmdbError> {
        let env = Environment::new()
            .set_map_size(10 * 1024 * 1024) // 10MB
            .open(path)?;
        let db = env.create_db(None, DatabaseFlags::empty())?;
        
        let mut hasher = VersionHasher { env, db };
        
        // Initialize main branch if it doesn't exist
        if !hasher.branch_exists("main")? {
            hasher.insert_branch("main", BranchInfo {
                branch_name: "main".to_string(),
                current_hash: String::new(),
                parent_branch: String::new(),
                parent_hash: String::new(),
                parent_version: 0,
            })?;
        }
        
        Ok(hasher)
    }
}

impl VersionHasher {
    fn branch_exists(&self, branch: &str) -> Result<bool, LmdbError> {
        let txn = self.env.begin_ro_txn()?;
        let exists = txn.get(self.db, &branch.to_string()).is_ok();
        txn.commit()?;
        Ok(exists)
    }
}

impl VersionHasher {
    fn insert_branch(&mut self, branch: &str, info: BranchInfo) -> Result<(), LmdbError> {
        let mut txn = self.env.begin_rw_txn()?;
        txn.put(self.db, &branch.to_string(), &bincode::serialize(&info).unwrap(), WriteFlags::empty())?;
        txn.commit()?;
        Ok(())
    }
}

impl VersionHasher {
    fn get_branch_info(&self, branch: &str) -> Result<Option<BranchInfo>, LmdbError> {
        let txn = self.env.begin_ro_txn()?;
        let result = txn.get(self.db, &branch.to_string())
            .map(|bytes| bincode::deserialize(bytes).unwrap());
        txn.commit()?;
        Ok(result.ok())
    }
}

impl VersionHasher {
    fn create_new_branch(&mut self, branch: &str, parent_branch: &str, parent_hash: String, parent_version: u64) -> Result<(), LmdbError> {
        let info = BranchInfo {
            branch_name: branch.to_string(),
            current_hash: String::new(),
            parent_branch: parent_branch.to_string(),
            parent_hash,
            parent_version,
        };
        self.insert_branch(branch, info)
    }
}

impl VersionHasher {
    fn trace_to_main(&self, start_branch: &str) -> Result<Vec<BranchInfo>, LmdbError> {
        let mut branch_path = Vec::new();
        let mut current_branch = start_branch.to_string();
        
        while current_branch != "main" {
            if let Some(info) = self.get_branch_info(&current_branch)? {
                branch_path.push(info.clone());
                current_branch = info.parent_branch;
            } else {
                return Err(LmdbError::NotFound);
            }
        }
        
        // Add the main branch info
        if let Some(main_info) = self.get_branch_info("main")? {
            branch_path.push(main_info);
        }
        
        branch_path.reverse();
        Ok(branch_path)
    }
}
