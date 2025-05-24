use std::sync::atomic::AtomicU32;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    lazy_item::ProbLazyItem,
    tf_idf_index::TFIDFIndexNodeData,
    types::FileOffset,
};

use super::TFIDFIndexSerialize;

impl TFIDFIndexSerialize for *mut ProbLazyItem<TFIDFIndexNodeData> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        let file_offset = lazy_item.file_index.offset.0;

        if let Some(data) = lazy_item.unsafe_get_data() {
            dim_bufman.seek_with_cursor(cursor, file_offset as u64)?;
            data.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        }

        Ok(file_offset)
    }

    fn deserialize(
        _dim_bufmans: &BufferManager,
        _data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        _data_file_parts: u8,
        cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        cache.get_data(file_offset, data_file_idx)
    }
}
