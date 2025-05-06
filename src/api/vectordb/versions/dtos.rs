use serde::Serialize;

use crate::models::versioning::{Timestamp, VersionHash, VersionNumber};

#[derive(Serialize)]
pub struct VersionMetadata {
    pub hash: VersionHash,
    pub version_number: VersionNumber,
    pub timestamp: Timestamp,
    pub vector_count: u64,
}

#[derive(Serialize)]
pub struct VersionListResponse {
    pub versions: Vec<VersionMetadata>,
    pub current_hash: VersionHash,
}

#[derive(Serialize)]
pub struct CurrentVersionResponse {
    pub hash: VersionHash,
    pub version_number: VersionNumber,
    pub timestamp: Timestamp,
    pub vector_count: u64,
}
