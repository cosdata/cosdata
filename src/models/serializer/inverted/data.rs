use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    common::TSHashTable,
    inverted_index::InvertedIndexNodeData,
    page::VersionedPagepool,
    types::FileOffset,
};

use super::InvertedIndexSerialize;

impl InvertedIndexSerialize for InvertedIndexNodeData {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start = dim_bufman.cursor_position(cursor)?;
        let data_bufman = data_bufmans.get(data_file_idx)?;
        let data_cursor = data_bufman.open_cursor()?;
        for i in 0..=self.max_key {
            let Some(pool) = self.map.lookup(&i) else {
                dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                continue;
            };
            let offset = pool.serialize(
                dim_bufman,
                data_bufmans,
                data_file_idx,
                data_file_parts,
                data_cursor,
            )?;
            dim_bufman.update_u32_with_cursor(cursor, offset)?;
        }
        Ok(start as u32)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        data_file_parts: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64 - 1)?;
        let quantization_and_implicit = dim_bufman.read_u8_with_cursor(cursor)?;
        let quantization_bits = (quantization_and_implicit << 1) >> 1;
        let max_key = ((1u32 << quantization_bits) - 1) as u8;

        let map = TSHashTable::new(16);
        for i in 0..=max_key {
            let offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                continue;
            }
            let pool = VersionedPagepool::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(offset),
                data_file_idx,
                data_file_parts,
                cache,
            )?;
            map.insert(i, pool);
        }
        dim_bufman.close_cursor(cursor)?;
        Ok(Self { map, max_key })
    }
}
