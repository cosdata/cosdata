use std::sync::{atomic::Ordering, Arc};

use crate::models::{
    common::WaCustomError,
    meta_persist::store_highest_internal_id,
    versioning::{Hash, Version},
};

use super::TFIDFIndex;

pub struct TFIDFIndexTransaction {
    pub id: Hash,
    pub version_number: u16,
}

impl TFIDFIndexTransaction {
    pub fn new(tf_idf_index: Arc<TFIDFIndex>) -> Result<Self, WaCustomError> {
        let branch_info = tf_idf_index
            .vcs
            .get_branch_info("main")
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get main branch info: {}", err))
            })?
            .unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = tf_idf_index
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

    pub fn pre_commit(self, tf_idf_index: &TFIDFIndex) -> Result<(), WaCustomError> {
        store_highest_internal_id(
            &tf_idf_index.lmdb,
            tf_idf_index.document_id_counter.load(Ordering::Relaxed),
        )?;
        tf_idf_index.vec_raw_map.serialize(
            &tf_idf_index.vec_raw_manager,
            tf_idf_index.root.data_file_parts,
        )?;
        tf_idf_index.root.serialize()?;
        tf_idf_index.root.cache.dim_bufman.flush()?;
        tf_idf_index.root.cache.data_bufmans.flush_all()?;
        Ok(())
    }
}
