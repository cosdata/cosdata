use crate::models::versioning::Hash;
use serde::Serialize;

#[derive(Serialize)]
pub struct VersionMetadata {
    pub hash: Hash,
    pub version_number: u32,
    pub timestamp: u32,
    pub vector_count: u64,
}

#[derive(Serialize)]
pub struct VersionListResponse {
    pub versions: Vec<VersionMetadata>,
    pub current_hash: Hash,
}

#[derive(Serialize)]
pub struct CurrentVersionResponse {
    pub hash: Hash,
    pub version_number: u32,
    pub timestamp: u64,
    pub vector_count: u64,
}
