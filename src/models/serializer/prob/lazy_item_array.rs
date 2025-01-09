use std::{
    collections::HashSet,
    io::{self, SeekFrom},
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::{FileIndex, SyncPersist},
    prob_lazy_load::lazy_item_array::ProbLazyItemArray,
    prob_node::{ProbNode, SharedNode},
    types::FileOffset,
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

impl<const N: usize> ProbSerialize for ProbLazyItemArray<ProbNode, N> {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)?;
        bufman.write_with_cursor(cursor, &vec![u8::MAX; 10 * N])?;

        for i in 0..N {
            let Some(item_ptr) = self.get(i) else {
                break;
            };
            bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

            let offset = item_ptr.serialize(bufmans, version, cursor)?;
            let placeholder_pos = start_offset + (i as u64 * 10);

            let item = unsafe { &*item_ptr };

            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            bufman.write_u16_with_cursor(cursor, item.get_current_version_number())?;
            bufman.write_u32_with_cursor(cursor, *item.get_current_version())?;
        }
        bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize ProbLazyItemArray with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;

                let placeholder_offset = offset as u64;
                let array = Self::new();

                for i in 0..N {
                    bufman.seek_with_cursor(
                        cursor,
                        SeekFrom::Start(placeholder_offset + (i as u64 * 10)),
                    )?;
                    let offset = bufman.read_u32_with_cursor(cursor)?;
                    if offset == u32::MAX {
                        break;
                    }
                    let version_number = bufman.read_u16_with_cursor(cursor)?;
                    let version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);
                    let file_index = FileIndex::Valid {
                        offset: FileOffset(offset),
                        version_number,
                        version_id,
                    };
                    let item =
                        SharedNode::deserialize(bufmans, file_index, cache, max_loads, skipm)?;
                    array.push(item);
                }

                bufman.close_cursor(cursor)?;

                Ok(array)
            }
        }
    }
}

impl<const N: usize> UpdateSerialized for ProbLazyItemArray<ProbNode, N> {
    fn update_serialized(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
    ) -> Result<u32, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot update ProbLazyItemArray with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                ..
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                let placeholder_start = offset as u64;
                let mut i = 0;

                for j in 0..N {
                    let placeholder_pos = placeholder_start + (j as u64 * 10);
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
                    let offset = bufman.read_u32_with_cursor(cursor)?;
                    if offset == u32::MAX {
                        break;
                    }
                    i += 1;
                }

                for j in i..N {
                    let Some(item_ptr) = self.get(j) else {
                        break;
                    };
                    bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
                    let offset = item_ptr.serialize(bufmans, version_id, cursor)?;
                    let placeholder_pos = placeholder_start + (j as u64 * 10);

                    let item = unsafe { &*item_ptr };

                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
                    bufman.write_u32_with_cursor(cursor, offset)?;
                    bufman.write_u16_with_cursor(cursor, item.get_current_version_number())?;
                    bufman.write_u32_with_cursor(cursor, *item.get_current_version())?;
                }
                bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

                Ok(offset)
            }
        }
    }
}
