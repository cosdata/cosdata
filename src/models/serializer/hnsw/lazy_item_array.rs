use std::collections::HashSet;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::HNSWIndexCache,
    prob_lazy_load::{lazy_item::FileIndex, lazy_item_array::ProbLazyItemArray},
    prob_node::{ProbNode, SharedNode},
    types::FileOffset,
    versioning::Hash,
};

use super::HNSWIndexSerialize;

// @SERIALIZED_SIZE:
//   length * (
//     4 bytes for item offset +
//     2 bytes for version number +
//     4 bytes for version hash
//   ) = N * 10
impl<const N: usize> HNSWIndexSerialize for ProbLazyItemArray<ProbNode, N> {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)?;
        bufman.update_with_cursor(cursor, &vec![u8::MAX; 10 * N])?;
        bufman.seek_with_cursor(cursor, start_offset)?;

        for i in 0..N {
            let Some(item_ptr) = self.get(i) else {
                break;
            };

            let item = unsafe { &*item_ptr };
            let FileIndex {
                offset,
                version_number,
                version_id,
            } = item.get_file_index();

            bufman.update_u32_with_cursor(cursor, offset.0)?;
            bufman.update_u16_with_cursor(cursor, version_number)?;
            bufman.update_u32_with_cursor(cursor, *version_id)?;
        }

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError> {
        let FileIndex {
            offset: FileOffset(offset),
            version_id,
            ..
        } = file_index;

        let bufman = bufmans.get(version_id)?;
        let cursor = bufman.open_cursor()?;

        let placeholder_offset = offset as u64;
        let array = Self::new();

        for i in 0..N {
            bufman.seek_with_cursor(cursor, placeholder_offset + (i as u64 * 10))?;
            let offset = bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                break;
            }
            let version_number = bufman.read_u16_with_cursor(cursor)?;
            let version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);
            let file_index = FileIndex {
                offset: FileOffset(offset),
                version_number,
                version_id,
            };
            let item =
                SharedNode::deserialize(bufmans, file_index, cache, max_loads, skipm, is_level_0)?;
            array.push(item);
        }

        bufman.close_cursor(cursor)?;

        Ok(array)
    }
}
