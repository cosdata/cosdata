use bincode::{deserialize, serialize};
use bs58;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher24;

#[derive(
    Serialize, Deserialize, Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct VersionHash {
    pub branch: String,
    pub version: u32,
    pub hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BranchInfo {
    current_hash: String,
    parent_branch: String,
    parent_hash: String,
    parent_version: u32,
}

use std::{collections::HashMap, hash::Hasher};

pub struct VersionHasher {
    branches: HashMap<String, BranchInfo>,
}

impl VersionHasher {
    pub fn new() -> Self {
        let mut branches = HashMap::new();
        branches.insert(
            "main".to_string(),
            BranchInfo {
                current_hash: String::new(),
                parent_branch: String::new(),
                parent_hash: String::new(),
                parent_version: 0,
            },
        );
        Self { branches }
    }

    pub fn generate_hash(
        &mut self,
        branch: &str,
        version: u32,
        parent_branch: Option<&str>,
        parent_version: Option<u32>,
    ) -> VersionHash {
        let (parent_hash, parent_branch, parent_version) =
            if let Some(branch_info) = self.branches.get(branch) {
                (
                    branch_info.current_hash.clone(),
                    branch_info.parent_branch.clone(),
                    branch_info.parent_version,
                )
            } else {
                let parent_branch = parent_branch.unwrap_or("main").to_string();
                let parent_info = self.branches.get(&parent_branch).unwrap();
                (
                    parent_info.current_hash.clone(),
                    parent_branch,
                    parent_version.unwrap_or(0),
                )
            };

        let input = format!("{}{}{}", parent_hash, branch, version);

        let mut hasher = SipHasher24::new();
        hasher.write(input.as_bytes());
        let num = hasher.finish();
        let bytes = num.to_be_bytes();
        let base58_str = bs58::encode(bytes).into_string();

        self.branches.insert(
            branch.to_string(),
            BranchInfo {
                current_hash: base58_str.clone(),
                parent_branch,
                parent_hash,
                parent_version,
            },
        );

        VersionHash {
            branch: branch.to_string(),
            version,
            hash: base58_str,
        }
    }
}
