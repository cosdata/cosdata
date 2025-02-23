use serde::{Deserialize, Serialize};
use crate::models::versioning::Hash;

#[derive(Serialize)]
pub struct VersionMetadata {
    pub hash: Hash,
    pub version_number: u32,
    pub timestamp: u64,
    pub vector_count: u64,
}

#[derive(Serialize)]
pub struct VersionListResponse {
    pub versions: Vec<VersionMetadata>,
    pub current_version: Hash,
}

#[derive(Serialize)]
pub struct CurrentVersionResponse {
    pub hash: Hash,
    pub version_number: u32,
    pub timestamp: u64,
    pub vector_count: u64,
}

