use std::{collections::HashSet, io::SeekFrom, sync::Arc};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::NodeRegistry,
    common::TSHashTable,
    lazy_load::FileIndex,
    types::FileOffset,
    versioning::Hash,
};

use super::CustomSerialize;

impl<V> CustomSerialize for TSHashTable<u8, V>
where
    V: CustomSerialize,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;

        let start_offset = bufman.cursor_position(cursor)? as u32;

        // Write the size of the list at first
        bufman.write_u8_with_cursor(cursor, self.size)?;

        bufman.write_with_cursor(cursor, &[u8::MAX; 256])?;

        for key in 0..64 {
            let placeholder = start_offset + 1 + (key as u32 * 4);
            bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
            let Some(offset) =
                self.with_value(&key, |v| v.serialize(bufmans.clone(), version, cursor))
            else {
                continue;
            };
            let offset = offset?;

            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder as u64))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
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
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;

                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                // Read the length of the vec
                let size = bufman.read_u8_with_cursor(cursor)?;

                let table = Self::new(size);

                for key in 0..64 {
                    let placeholder_offset = offset + 1 + (key as u32 * 4);
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_offset as u64))?;
                    let value_offset = bufman.read_u32_with_cursor(cursor)?;
                    if value_offset == u32::MAX {
                        continue;
                    }
                    let value_file_index = FileIndex::Valid {
                        offset: FileOffset(value_offset),
                        version_number,
                        version_id,
                    };
                    let value = V::deserialize(
                        bufmans.clone(),
                        value_file_index,
                        cache.clone(),
                        max_loads,
                        skipm,
                    )?;
                    table.insert(key, value);
                }

                bufman.close_cursor(cursor)?;

                Ok(table)
            }

            FileIndex::Invalid => Err(BufIoError::Locking),
        }
    }
}
