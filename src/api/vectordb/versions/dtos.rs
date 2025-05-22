use serde::Serialize;

use crate::models::versioning::VersionNumber;

#[derive(Serialize)]
pub struct VersionMetadata {
    pub version_number: VersionNumber,
    pub vector_count: u64,
}

#[derive(Serialize)]
pub struct VersionListResponse {
    pub versions: Vec<VersionMetadata>,
    pub current_version: VersionNumber,
}

#[derive(Serialize)]
pub struct CurrentVersionResponse {
    pub version_number: VersionNumber,
    pub vector_count: u64,
}
