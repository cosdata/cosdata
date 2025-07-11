use std::sync::{atomic::AtomicU32, RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    serializer::SimpleSerialize,
    tf_idf_index::TermInfo,
    types::FileOffset,
    versioned_vec::VersionedVec,
};

use super::TFIDFIndexSerialize;

impl TFIDFIndexSerialize for TermInfo {
    fn serialize(
        &self,
        _dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        _offset_counter: &AtomicU32,
        data_file_idx: u8,
        _data_file_parts: u8,
        _cursor: u64,
    ) -> Result<u32, BufIoError> {
        let data_bufman = data_bufmans.get(data_file_idx)?;
        let data_cursor = data_bufman.open_cursor()?;

        let offset = self
            .documents
            .read()
            .map_err(|_| BufIoError::Locking)?
            .serialize(&data_bufman, data_cursor)?;

        data_bufman.close_cursor(data_cursor)?;
        Ok(offset)
    }

    fn deserialize(
        _dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        _data_file_parts: u8,
        _cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        let data_bufman = data_bufmans.get(data_file_idx)?;
        let documents = VersionedVec::deserialize(&data_bufman, file_offset)?;

        Ok(Self {
            documents: RwLock::new(documents),
            sequence_idx: 0, // Handled by caller
        })
    }
}
