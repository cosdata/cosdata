use super::CustomSerialize;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::{Cacheable, NodeRegistry},
    identity_collections::{Identifiable, IdentitySet},
    lazy_load::{FileIndex, LazyItem, LazyItemSet, SyncPersist, CHUNK_SIZE},
    types::FileOffset,
    versioning::Hash,
};
use std::collections::HashSet;
use std::{io::SeekFrom, sync::Arc};

impl<T> CustomSerialize for LazyItemSet<T>
where
    T: Cacheable + Clone + Identifiable<Id = u64> + CustomSerialize + 'static,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        if self.is_empty() {
            return Ok(u32::MAX);
        };
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;
        let mut items_arc = self.items.clone();
        let items: Vec<_> = items_arc.get().iter().map(Clone::clone).collect();
        let total_items = items.len();

        for chunk_start in (0..total_items).step_by(CHUNK_SIZE) {
            let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, total_items);
            let is_last_chunk = chunk_end == total_items;

            // Write placeholders for item offsets
            let placeholder_start = bufman.cursor_position(cursor)? as u32;
            for _ in 0..CHUNK_SIZE {
                bufman.write_u32_with_cursor(cursor, u32::MAX)?;
                bufman.write_u16_with_cursor(cursor, u16::MAX)?;
                bufman.write_u32_with_cursor(cursor, u32::MAX)?;
            }
            // Write placeholder for next chunk link
            let next_chunk_placeholder = bufman.cursor_position(cursor)? as u32;
            bufman.write_u32_with_cursor(cursor, u32::MAX)?;

            // Serialize items and update placeholders
            for i in chunk_start..chunk_end {
                let item_offset = items[i].serialize(bufmans.clone(), version, cursor)?;
                let placeholder_pos = placeholder_start as u64 + ((i - chunk_start) as u64 * 10);
                let current_pos = bufman.cursor_position(cursor)?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
                bufman.write_u32_with_cursor(cursor, item_offset)?;
                bufman.write_u16_with_cursor(cursor, items[i].get_current_version_number())?;
                bufman.write_u32_with_cursor(cursor, *items[i].get_current_version())?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(current_pos))?;
            }

            // Write next chunk link
            let next_chunk_start = bufman.cursor_position(cursor)? as u32;
            bufman.seek_with_cursor(cursor, SeekFrom::Start(next_chunk_placeholder as u64))?;
            if is_last_chunk {
                bufman.write_u32_with_cursor(cursor, u32::MAX)?; // Last chunk
            } else {
                bufman.write_u32_with_cursor(cursor, next_chunk_start)?;
            }
            bufman.seek_with_cursor(cursor, SeekFrom::Start(next_chunk_start as u64))?;
        }
        Ok(start_offset)
    }
    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Ok(LazyItemSet::new()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                if offset == u32::MAX {
                    return Ok(LazyItemSet::new());
                }
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                let mut items = Vec::new();
                let mut current_chunk = offset;
                loop {
                    for i in 0..CHUNK_SIZE {
                        bufman.seek_with_cursor(
                            cursor,
                            SeekFrom::Start(current_chunk as u64 + (i as u64 * 10)),
                        )?;
                        let item_offset = bufman.read_u32_with_cursor(cursor)?;
                        let item_version_number = bufman.read_u16_with_cursor(cursor)?;
                        let item_version_id = bufman.read_u32_with_cursor(cursor)?.into();
                        if item_offset == u32::MAX {
                            continue;
                        }
                        let item_file_index = FileIndex::Valid {
                            offset: FileOffset(item_offset),
                            version_number: item_version_number,
                            version_id: item_version_id,
                        };
                        let item = LazyItem::deserialize(
                            bufmans.clone(),
                            item_file_index,
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?;
                        items.push(item);
                    }
                    bufman.seek_with_cursor(
                        cursor,
                        SeekFrom::Start(current_chunk as u64 + CHUNK_SIZE as u64 * 10),
                    )?;
                    // Read next chunk link
                    current_chunk = bufman.read_u32_with_cursor(cursor)?;
                    if current_chunk == u32::MAX {
                        break;
                    }
                }
                bufman.close_cursor(cursor)?;
                Ok(LazyItemSet::from_set(IdentitySet::from_iter(
                    items.into_iter(),
                )))
            }
        }
    }
}
