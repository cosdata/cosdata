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

use super::ProbSerialize;

// @SERIALIZED_SIZE:
//   length * (
//     4 bytes for item offset +
//     2 bytes for version number +
//     4 bytes for version hash
//   ) = N * 10
impl<const N: usize> ProbSerialize for ProbLazyItemArray<ProbNode, N> {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        level_0_bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
        direct: bool,
        is_level_0: bool,
    ) -> Result<u32, BufIoError> {
        let bufman = if is_level_0 {
            level_0_bufmans.get(version)?
        } else {
            bufmans.get(version)?
        };
        let start_offset = bufman.cursor_position(cursor)?;

        for i in 0..N {
            let Some(item_ptr) = self.get(i) else {
                break;
            };

            let offset = item_ptr.serialize(
                bufmans,
                level_0_bufmans,
                version,
                cursor,
                direct,
                is_level_0,
            )?;
            let placeholder_pos = start_offset + (i as u64 * 10);

            let item = unsafe { &*item_ptr };

            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            bufman.write_u16_with_cursor(cursor, item.get_current_version_number())?;
            bufman.write_u32_with_cursor(cursor, *item.get_current_version())?;
        }

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        level_0_bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
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
                let bufman = if is_level_0 {
                    level_0_bufmans.get(version_id)?
                } else {
                    bufmans.get(version_id)?
                };
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
                    let item = SharedNode::deserialize(
                        bufmans,
                        level_0_bufmans,
                        file_index,
                        cache,
                        max_loads,
                        skipm,
                        is_level_0,
                    )?;
                    array.push(item);
                }

                bufman.close_cursor(cursor)?;

                Ok(array)
            }
        }
    }
}
