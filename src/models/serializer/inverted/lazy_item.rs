use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    inverted_index::InvertedIndexNodeData,
    lazy_item::LazyItem,
    types::FileOffset,
};

use super::InvertedIndexSerialize;

impl InvertedIndexSerialize for *mut LazyItem<InvertedIndexNodeData> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
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
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        cache.get_data(file_offset, data_file_idx)
    }
}
