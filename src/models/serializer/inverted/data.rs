use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    common::TSHashTable,
    inverted_index::InvertedIndexNodeData,
    // serializer::SimpleSerialize,
    types::FileOffset,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

use super::InvertedIndexSerialize;

impl InvertedIndexSerialize for InvertedIndexNodeData {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start = dim_bufman.cursor_position(cursor)?;
        for i in 0..=self.max_key {
            let optional_result = self.map.with_value(&i, |vec| {
                let offset = vec.serialize(dim_bufman, data_bufmans, cursor)?;
                dim_bufman.update_u32_with_cursor(cursor, offset)?;
                dim_bufman.update_u32_with_cursor(cursor, *vec.version)
            });

            if let Some(result) = optional_result {
                result?;
            } else {
                dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                continue;
            };
        }
        Ok(start as u32)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        _version: VersionNumber,
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
            let version = dim_bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                continue;
            }
            let pool = VersionedVec::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(offset),
                VersionNumber::from(version),
                cache,
            )?;
            map.insert(i, pool);
        }
        dim_bufman.close_cursor(cursor)?;
        Ok(Self { map, max_key })
    }
}
