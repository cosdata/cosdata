use std::sync::atomic::AtomicU32;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    lazy_item::LazyItem,
    tf_idf_index::TFIDFIndexNodeData,
    types::FileOffset,
    versioning::VersionNumber,
};

use super::TFIDFIndexSerialize;

impl TFIDFIndexSerialize for *mut LazyItem<TFIDFIndexNodeData, ()> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        offset_counter: &AtomicU32,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        let file_offset = lazy_item.file_index.offset.0;

        if let Some(data) = lazy_item.unsafe_get_data() {
            dim_bufman.seek_with_cursor(cursor, file_offset as u64)?;
            data.serialize(dim_bufman, data_bufmans, offset_counter, cursor)?;
        }

        Ok(file_offset)
    }

    fn deserialize(
        _dim_bufmans: &BufferManager,
        _data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        _version: VersionNumber,
        cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        cache.get_data(file_offset)
    }
}
