use std::{collections::HashSet, io::SeekFrom, sync::Arc};

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::NodeRegistry,
        lazy_load::FileIndex,
        types::FileOffset,
        versioning::Hash,
    },
    storage::page::{Page, Pagepool},
};

use super::CustomSerialize;

impl<const LEN: usize> CustomSerialize for Pagepool<LEN> {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let total_items = self.inner.len();
        if total_items == 0 {
            return Ok(u32::MAX);
        }
        let bufman = bufmans.get(version)?;

        // Move the cursor to the end of the file and start writing from there
        let start_offset = if let Some(offset) = self.inner[0].serialized_at.get() {
            bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
            offset
        } else {
            let offset = bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;
            self.inner[0].serialized_at.set(Some(offset));
            offset
        };

        bufman.write_u32_with_cursor(cursor, u32::MAX)?;
        self.inner[0].serialize(bufmans.clone(), version, cursor)?;

        let mut prev_offset = start_offset;

        for item in self.inner.iter().skip(1) {
            let offset = if let Some(offset) = item.serialized_at.get() {
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                offset
            } else {
                let offset = bufman.seek_with_cursor(cursor, SeekFrom::End(0))? as u32;
                item.serialized_at.set(Some(offset));
                offset
            };
            bufman.write_u32_with_cursor(cursor, u32::MAX)?;
            item.serialize(bufmans.clone(), version, cursor)?;

            let current_offset = bufman.cursor_position(cursor)?;

            bufman.seek_with_cursor(cursor, SeekFrom::Start(prev_offset as u64))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            bufman.seek_with_cursor(cursor, SeekFrom::Start(current_offset))?;

            prev_offset = offset;
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
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                version_number,
            } => {
                if offset == u32::MAX {
                    return Ok(Self::default());
                }
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;

                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                let mut page_pool = Pagepool::<LEN>::default();
                let mut current_chunk_offset = offset;
                loop {
                    let next_chunk_offset = bufman.read_u32_with_cursor(cursor)?;
                    let item_file_index = FileIndex::Valid {
                        offset: FileOffset(current_chunk_offset + 4),
                        version_id,
                        version_number,
                    };
                    let item = Page::deserialize(
                        bufmans.clone(),
                        item_file_index,
                        cache.clone(),
                        max_loads,
                        skipm,
                    )?;

                    page_pool.inner.push(item);

                    if next_chunk_offset == u32::MAX {
                        break;
                    }
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(next_chunk_offset as u64))?;
                    current_chunk_offset = next_chunk_offset;
                }

                bufman.close_cursor(cursor)?;
                Ok(page_pool)
            }
            FileIndex::Invalid => Ok(Pagepool::default()),
        }
    }
}

impl<const LEN: usize> CustomSerialize for Page<LEN> {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;

        bufman.write_u32_with_cursor(cursor, self.len as u32)?;
        for item in self.data {
            bufman.write_u32_with_cursor(cursor, item)?;
        }

        Ok(start_offset)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        _cache: Arc<NodeRegistry>,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                let len = bufman.read_u32_with_cursor(cursor)?;

                let mut items = Self::new();

                for _ in 0..len {
                    items.push(bufman.read_u32_with_cursor(cursor)?)
                }

                // starting from the next chunk placeholder
                items.serialized_at.set(Some(offset - 4));

                Ok(items)
            }

            FileIndex::Invalid => Ok(Self::new()),
        }
    }
}
