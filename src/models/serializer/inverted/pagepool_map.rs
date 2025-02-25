use std::collections::HashSet;

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::InvertedIndexCache,
        common::TSHashTable,
        types::FileOffset,
        versioning::Hash,
    },
    storage::page::Pagepool,
};

use super::{InvertedIndexFileIndex, InvertedIndexSerialize};

impl<const LEN: usize> InvertedIndexSerialize<(Hash, u8, u8)> for TSHashTable<u8, Pagepool<LEN>> {
    fn serialize(
        &self,
        dim_bufmans: &BufferManagerFactory<Hash>,
        data_bufmans: &BufferManagerFactory<(Hash, u8)>,
        (version, data_file_idx, quantization): (Hash, u8, u8),
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = dim_bufmans.get(version)?;
        let start = bufman.cursor_position(cursor)?;
        let data_bufman = data_bufmans.get((version, data_file_idx))?;
        let data_cursor = data_bufman.open_cursor()?;
        for i in 0..quantization {
            let Some(pool) = self.lookup(&i) else {
                bufman.update_u32_with_cursor(cursor, u32::MAX)?;
                continue;
            };
            let offset = pool.serialize(
                dim_bufmans,
                data_bufmans,
                (version, data_file_idx),
                data_cursor,
            )?;
            bufman.update_u32_with_cursor(cursor, offset)?;
        }
        Ok(start as u32)
    }

    fn deserialize(
        dim_bufmans: &BufferManagerFactory<Hash>,
        data_bufmans: &BufferManagerFactory<(Hash, u8)>,
        file_index: InvertedIndexFileIndex<(Hash, u8, u8)>,
        cache: &InvertedIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let map = Self::new(16);
        let bufman = dim_bufmans.get(file_index.file_identifier.0)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_index.offset.0 as u64)?;
        for i in 0..file_index.file_identifier.2 {
            let offset = bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                continue;
            }
            let pagepool_file_index = InvertedIndexFileIndex {
                offset: FileOffset(offset),
                version_number: file_index.version_number,
                file_identifier: (file_index.file_identifier.0, file_index.file_identifier.1),
            };
            let pool = Pagepool::deserialize(
                dim_bufmans,
                data_bufmans,
                pagepool_file_index,
                cache,
                max_loads,
                skipm,
            )?;
            map.insert(i, pool);
        }
        bufman.close_cursor(cursor)?;
        Ok(map)
    }
}
