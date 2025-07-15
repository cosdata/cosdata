use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::models::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    lazy_item::LazyItem,
    tf_idf_index::{TFIDFIndexNode, TFIDFIndexNodeData},
    types::FileOffset,
};

use super::{TFIDFIndexSerialize, TF_IDF_INDEX_DATA_CHUNK_SIZE};

// @SERIALIZED_SIZE:
//
//   4 byte for dim index +                          | 4
//   2 bytes for data map len +                      | 6
//   INVERTED_INDEX_DATA_CHUNK_SIZE * (              |
//     2 bytes for quotient +                        |
//     4 bytes of versioned vec                      |
//   ) +                                             | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 6
//   4 byte for next data chunk                      | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 10
//   16 * 4 bytes for dimension offsets +            | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 74
impl TFIDFIndexSerialize for TFIDFIndexNode {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        _: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let data_file_idx = (self.dim_index % data_file_parts as u32) as u8;
        if !self.is_serialized.swap(true, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64)?;
            dim_bufman.update_u32_with_cursor(cursor, self.dim_index)?;
            self.data.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 10,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        } else if self.is_dirty.swap(false, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64 + 4)?;
            self.data.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 10,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        } else {
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 10,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        };
        Ok(self.file_offset.0)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        _: u8,
        data_file_parts: u8,
        cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let dim_index = dim_bufman.read_u32_with_cursor(cursor)?;
        let data_file_idx = (dim_index % data_file_parts as u32) as u8;
        let data = <*mut LazyItem<TFIDFIndexNodeData, ()>>::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 4),
            data_file_idx,
            data_file_parts,
            cache,
        )?;
        let children = AtomicArray::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 10),
            data_file_idx,
            data_file_parts,
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
