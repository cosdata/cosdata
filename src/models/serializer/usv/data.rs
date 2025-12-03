use std::sync::{
    atomic::{AtomicU16, AtomicU32, Ordering},
    Arc,
};

use parking_lot::RwLock;

use super::{USVIndexSerialize, USV_INDEX_DATA_CHUNK_SIZE};
use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::USVIndexCache,
    common::TSHashTable,
    types::FileOffset,
    usv_index::{IdLists, USVIndexNodeData},
    versioning::VersionNumber,
};

impl USVIndexSerialize for USVIndexNodeData {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        quantization_bits: u8,
        offset_counter: &AtomicU32,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start = dim_bufman.cursor_position(cursor)? as u32;
        let mut list = self.map.to_list();
        let mut num_entries_serialized = self.num_entries_serialized.write();
        let map_len = self.map_len.load(Ordering::Acquire);
        debug_assert_eq!(map_len, list.len() as u16);
        list.sort_unstable_by_key(|(_, v)| v.sequence_idx);
        dim_bufman.update_u16_with_cursor(cursor, map_len)?;

        let total_chunks = list.len().div_ceil(USV_INDEX_DATA_CHUNK_SIZE);

        for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * USV_INDEX_DATA_CHUNK_SIZE)
                ..((chunk_idx + 1) * USV_INDEX_DATA_CHUNK_SIZE)
            {
                let current_offset = dim_bufman.cursor_position(cursor)?;
                if *num_entries_serialized > i as u16 {
                    list[i].1.serialize(
                        dim_bufman,
                        data_bufmans,
                        quantization_bits,
                        offset_counter,
                        cursor,
                    )?;
                    dim_bufman.seek_with_cursor(cursor, current_offset + 6)?;
                } else if let Some((quotient, id_list)) = list.get(i) {
                    let offset = id_list.serialize(
                        dim_bufman,
                        data_bufmans,
                        quantization_bits,
                        offset_counter,
                        cursor,
                    )?;
                    dim_bufman.seek_with_cursor(cursor, current_offset)?;
                    dim_bufman.update_u16_with_cursor(cursor, *quotient)?;
                    dim_bufman.update_u32_with_cursor(cursor, offset)?;
                } else {
                    dim_bufman.update_with_cursor(cursor, &[u8::MAX; 6])?;
                }
            }
            if *num_entries_serialized > ((chunk_idx + 1) * USV_INDEX_DATA_CHUNK_SIZE) as u16 {
                let offset = dim_bufman.read_u32_with_cursor(cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset as u64)?;
            } else if chunk_idx == total_chunks - 1 {
                dim_bufman.update_with_cursor(cursor, &[u8::MAX; 4])?;
            } else {
                let offset = offset_counter.fetch_add(
                    (6 * USV_INDEX_DATA_CHUNK_SIZE + 4) as u32,
                    Ordering::Relaxed,
                );
                dim_bufman.update_u32_with_cursor(cursor, offset)?;
                dim_bufman.seek_with_cursor(cursor, offset as u64)?;
            }
        }

        *num_entries_serialized = list.len() as u16;

        Ok(start)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        _version: VersionNumber,
        cache: &USVIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let map_len = dim_bufman.read_u16_with_cursor(cursor)? as usize;
        let map = TSHashTable::new(16);

        let total_chunks = map_len.div_ceil(USV_INDEX_DATA_CHUNK_SIZE);

        for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * USV_INDEX_DATA_CHUNK_SIZE)
                ..((chunk_idx + 1) * USV_INDEX_DATA_CHUNK_SIZE)
            {
                if i == map_len {
                    break;
                }
                let quotient = dim_bufman.read_u16_with_cursor(cursor)?;
                let term_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                let mut term = IdLists::deserialize(
                    dim_bufman,
                    data_bufmans,
                    FileOffset(term_offset),
                    VersionNumber::from(u32::MAX), // not used
                    cache,
                )?;
                term.sequence_idx = i as u16;
                map.insert(quotient, Arc::new(term));
            }

            let next_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if next_chunk_offset == u32::MAX {
                break;
            }
            dim_bufman.seek_with_cursor(cursor, next_chunk_offset as u64)?;
        }

        Ok(Self {
            map,
            map_len: AtomicU16::new(map_len as u16),
            num_entries_serialized: RwLock::new(map_len as u16),
        })
    }
}
