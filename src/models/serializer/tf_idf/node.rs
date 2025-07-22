use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::models::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    lazy_item::LazyItem,
    tf_idf_index::{TFIDFIndexNode, TFIDFIndexNodeData},
    types::FileOffset,
    versioning::VersionNumber,
};

use super::{TFIDFIndexSerialize, TF_IDF_INDEX_DATA_CHUNK_SIZE};

// @SERIALIZED_SIZE:
//
//   4 byte for dim index +                          | 4
//   2 bytes for data map len +                      | 6
//   INVERTED_INDEX_DATA_CHUNK_SIZE * (              |
//     2 bytes for quotient +                        |
//     4 + 4 bytes of versioned vec offset & version |
//   ) +                                             | INVERTED_INDEX_DATA_CHUNK_SIZE * 10 + 6
//   4 byte for next data chunk                      | INVERTED_INDEX_DATA_CHUNK_SIZE * 10 + 10
//   16 * 4 bytes for dimension offsets +            | INVERTED_INDEX_DATA_CHUNK_SIZE * 10 + 74
impl TFIDFIndexSerialize for TFIDFIndexNode {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        offset_counter: &AtomicU32,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        if !self.is_serialized.swap(true, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64)?;
            dim_bufman.update_u32_with_cursor(cursor, self.dim_index)?;
            self.data
                .serialize(dim_bufman, data_bufmans, offset_counter, cursor)?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 10 + 10,
            )?;
            self.children
                .serialize(dim_bufman, data_bufmans, offset_counter, cursor)?;
        } else if self.is_dirty.swap(false, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64 + 4)?;
            self.data
                .serialize(dim_bufman, data_bufmans, offset_counter, cursor)?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 10 + 10,
            )?;
            self.children
                .serialize(dim_bufman, data_bufmans, offset_counter, cursor)?;
        } else {
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 10 + 10,
            )?;
            self.children
                .serialize(dim_bufman, data_bufmans, offset_counter, cursor)?;
        };
        Ok(self.file_offset.0)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
        cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let dim_index = dim_bufman.read_u32_with_cursor(cursor)?;
        let data = <*mut LazyItem<TFIDFIndexNodeData, ()>>::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 4),
            version,
            cache,
        )?;
        let children = AtomicArray::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u32 * 10 + 10),
            version,
            cache,
        )?;

        Ok(Self {
            is_serialized: AtomicBool::new(true),
            is_dirty: AtomicBool::new(false),
            dim_index,
            file_offset,
            data,
            children,
        })
    }
}
