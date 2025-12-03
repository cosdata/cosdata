use std::sync::atomic::AtomicU32;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::USVIndexCache,
    common::TSHashTable,
    types::FileOffset,
    usv_index::IdLists,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

use super::USVIndexSerialize;

impl USVIndexSerialize for IdLists {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        quantization_bits: u8,
        offset_counter: &AtomicU32,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let prev_offset = dim_bufman.cursor_position(cursor)?;
        dim_bufman.seek_with_cursor(cursor, self.offset.0 as u64)?;

        let max_key = ((1u16 << quantization_bits) - 1) as u8;

        for key in 0..=max_key {
            let result = self.map.with_value(&key, |v| {
                let offset = v.serialize(
                    dim_bufman,
                    data_bufmans,
                    quantization_bits,
                    offset_counter,
                    cursor,
                )?;
                let version = v.version;

                dim_bufman.update_u32_with_cursor(cursor, offset)?;
                dim_bufman.update_u32_with_cursor(cursor, *version)
            });

            if let Some(result) = result {
                result?;
            } else {
                dim_bufman.update_u64_with_cursor(cursor, u64::MAX)?;
            }
        }

        dim_bufman.seek_with_cursor(cursor, prev_offset)?;

        Ok(self.offset.0)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        _version: VersionNumber,
        cache: &USVIndexCache,
    ) -> Result<Self, BufIoError> {
        let map = TSHashTable::new(16);

        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;

        let max_key = ((1u16 << cache.quantization_bits) - 1) as u8;

        for key in 0..=max_key {
            let offset = dim_bufman.read_u32_with_cursor(cursor)?;
            let version = dim_bufman.read_u32_with_cursor(cursor)?;

            if offset == u32::MAX {
                continue;
            }

            let vec = VersionedVec::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(offset),
                version.into(),
                cache,
            )?;

            map.insert(key, vec);
        }

        Ok(Self {
            offset: file_offset,
            map,
            sequence_idx: 0,
        })
    }
}
