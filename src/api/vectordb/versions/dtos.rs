use serde::Serialize;

use crate::models::versioning::VersionNumber;

#[derive(Serialize, utoipa::ToSchema)]
pub struct VersionMetadata {
    pub version_number: VersionNumber,
    pub vector_count: u64,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct VersionListResponse {
    pub versions: Vec<VersionMetadata>,
    pub current_version: VersionNumber,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct CurrentVersionResponse {
    pub version_number: VersionNumber,
    pub vector_count: u64,
}
