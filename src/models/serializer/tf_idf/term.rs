use std::sync::{atomic::AtomicU32, RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    serializer::tree_map::TreeMapSerialize,
    tf_idf_index::TermInfo,
    types::FileOffset,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

use super::TFIDFIndexSerialize;

impl TFIDFIndexSerialize for TermInfo {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        _offset_counter: &AtomicU32,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let offset = self
            .documents
            .read()
            .map_err(|_| BufIoError::Locking)?
            .serialize(dim_bufman, data_bufmans, cursor)?;
        Ok(offset)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
        _cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        let documents = VersionedVec::deserialize(dim_bufman, data_bufmans, file_offset, version)?;

        Ok(Self {
            documents: RwLock::new(documents),
            sequence_idx: 0, // Handled by caller
        })
    }
}
