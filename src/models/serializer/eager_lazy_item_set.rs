use arcshift::ArcShift;

use super::CustomSerialize;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::{Cacheable, NodeRegistry},
    identity_collections::{Identifiable, IdentitySet},
    lazy_load::{EagerLazyItem, EagerLazyItemSet, FileIndex, CHUNK_SIZE},
    types::{FileOffset, STM},
    versioning::Hash,
};
use std::collections::HashSet;
use std::{io::SeekFrom, sync::Arc};

impl<T, E> CustomSerialize for EagerLazyItemSet<T, E>
where
    T: Cacheable + CustomSerialize + Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        if self.is_empty() {
            self.serialized_offset.clone().update(Some(u32::MAX));
            return Ok(u32::MAX);
        };
        let bufman = bufmans.get(version)?;
        let mut items_arc = self.items.clone();
        let items: Vec<_> = items_arc.get().iter().cloned().collect();
        let total_items = items.len();
        let first_serialize = if let Some(offset) = *self.serialized_offset.clone().get() {
            if offset == u32::MAX {
                bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;
                true
            } else {
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                false
            }
        } else {
            true
        };
        let start_offset = bufman.cursor_position(cursor)? as u32;

        if first_serialize {
            self.serialized_offset.clone().update(Some(start_offset));
        }
        for chunk_start in (0..total_items).step_by(CHUNK_SIZE) {
            let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, total_items);
            let is_last_chunk = chunk_end == total_items;

            // Write placeholders for item offsets
            let placeholder_start = bufman.cursor_position(cursor)? as u32;
            for _ in 0..CHUNK_SIZE {
                bufman.write_u32_with_cursor(cursor, u32::MAX)?;
            }
            // Write placeholder for next chunk link
            let next_chunk_placeholder = bufman.cursor_position(cursor)? as u32;
            bufman.write_u32_with_cursor(cursor, u32::MAX)?;

            // Serialize items and update placeholders
            for i in chunk_start..chunk_end {
                if !first_serialize {
                    bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
                }
                let item_offset = items[i].serialize(bufmans.clone(), version, cursor)?;
                let placeholder_pos = placeholder_start as u64 + ((i - chunk_start) as u64 * 4);
                let current_pos = bufman.cursor_position(cursor)?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
                bufman.write_u32_with_cursor(cursor, item_offset)?;
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
            FileIndex::Invalid => Ok(EagerLazyItemSet::new()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_number,
                version_id,
            } => {
                if offset == u32::MAX {
                    return Ok(EagerLazyItemSet::new());
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
                            SeekFrom::Start(current_chunk as u64 + (i as u64 * 4)),
                        )?;
                        let item_offset = bufman.read_u32_with_cursor(cursor)?;
                        if item_offset == u32::MAX {
                            continue;
                        }
                        let item_file_index = FileIndex::Valid {
                            offset: FileOffset(item_offset),
                            version_number,
                            version_id,
                        };
                        let item = EagerLazyItem::deserialize(
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
                        SeekFrom::Start(current_chunk as u64 + CHUNK_SIZE as u64 * 4),
                    )?;
                    // Read next chunk link
                    current_chunk = bufman.read_u32_with_cursor(cursor)?;
                    if current_chunk == u32::MAX {
                        break;
                    }
                }
                Ok(EagerLazyItemSet {
                    serialized_offset: ArcShift::new(Some(offset)),
                    items: STM::new(IdentitySet::from_iter(items.into_iter()), 5, false),
                })
            }
        }
    }
}
