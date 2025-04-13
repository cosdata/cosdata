use std::sync::Arc;

use crate::models::{
    common::WaCustomError,
    versioning::{Hash, Version},
};

use super::InvertedIndex;

pub struct InvertedIndexTransaction {
    pub id: Hash,
    pub version_number: u16,
}

impl InvertedIndexTransaction {
    pub fn new(inverted_index: Arc<InvertedIndex>) -> Result<Self, WaCustomError> {
        let branch_info = inverted_index
            .vcs
            .get_branch_info("main")
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get main branch info: {}", err))
            })?
            .unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = inverted_index
            .vcs
            .generate_hash("main", Version::from(version_number))
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get transaction hash: {}", err))
            })?;

        Ok(Self {
            id,
            version_number: version_number as u16,
        })
    }

    pub fn pre_commit(self, inverted_index: &InvertedIndex) -> Result<(), WaCustomError> {
        inverted_index.vec_raw_map.serialize(
            &inverted_index.vec_raw_manager,
            inverted_index.root.data_file_parts,
        )?;
        inverted_index.root.serialize()?;
        inverted_index.root.cache.dim_bufman.flush()?;
        inverted_index.root.cache.data_bufmans.flush_all()?;
        Ok(())
    }
}
