use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    common::TSHashTable,
    inverted_index::InvertedIndexNodeData,
    serializer::SimpleSerialize,
    tf_idf_index::VersionedVec,
    types::FileOffset,
};

use super::InvertedIndexSerialize;

impl InvertedIndexSerialize for InvertedIndexNodeData {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        _data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start = dim_bufman.cursor_position(cursor)?;
        let data_bufman = data_bufmans.get(data_file_idx)?;
        let data_cursor = data_bufman.open_cursor()?;
        for i in 0..=self.max_key {
            let optional_result = self.map.with_value(&i, |pool| {
                let offset = pool.serialize(&data_bufman, data_cursor)?;
                dim_bufman.update_u32_with_cursor(cursor, offset)
            });

            if let Some(result) = optional_result {
                result?;
            } else {
                dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                continue;
            };
        }
        Ok(start as u32)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        _data_file_parts: u8,
        _cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64 - 1)?;
        let quantization_and_implicit = dim_bufman.read_u8_with_cursor(cursor)?;
        let quantization_bits = (quantization_and_implicit << 1) >> 1;
        let max_key = ((1u32 << quantization_bits) - 1) as u8;
        let data_bufman = data_bufmans.get(data_file_idx)?;

        let map = TSHashTable::new(16);
        for i in 0..=max_key {
            let offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                continue;
            }
            let pool = VersionedVec::deserialize(&data_bufman, FileOffset(offset))?;
            map.insert(i, pool);
        }
        dim_bufman.close_cursor(cursor)?;
        Ok(Self { map, max_key })
    }
}
