use std::sync::Arc;

use crate::models::{
    common::WaCustomError,
    versioning::{Hash, Version},
};

use super::InvertedIndexIDF;

pub struct InvertedIndexIDFTransaction {
    pub id: Hash,
    pub version_number: u16,
}

impl InvertedIndexIDFTransaction {
    pub fn new(idf_inverted_index: Arc<InvertedIndexIDF>) -> Result<Self, WaCustomError> {
        let branch_info = idf_inverted_index
            .vcs
            .get_branch_info("main")
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get main branch info: {}", err))
            })?
            .unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = idf_inverted_index
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

    pub fn pre_commit(
        self,
        idf_inverted_index: Arc<InvertedIndexIDF>,
    ) -> Result<(), WaCustomError> {
        idf_inverted_index.vec_raw_map.serialize(
            &idf_inverted_index.vec_raw_manager,
            idf_inverted_index.root.data_file_parts,
        )?;
        idf_inverted_index.root.serialize()?;
        idf_inverted_index.root.cache.dim_bufman.flush()?;
        idf_inverted_index.root.cache.data_bufmans.flush_all()?;
        Ok(())
    }
}
