use crate::{
    models::{
        atomic_array::AtomicArray,
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        cache_loader::InvertedIndexCache,
        types::FileOffset,
    },
    storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasicTSHashmap,
};

use super::InvertedIndexSerialize;

impl<const N: usize> InvertedIndexSerialize
    for AtomicArray<InvertedIndexSparseAnnNodeBasicTSHashmap, N>
{
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start = dim_bufman.cursor_position(cursor)?;
        dim_bufman.update_with_cursor(cursor, &vec![u8::MAX; 4 * N])?;
        dim_bufman.seek_with_cursor(cursor, start)?;

        for i in 0..N {
            let Some(item_ptr) = self.get(i) else {
                dim_bufman.update_with_cursor(cursor, &[u8::MAX; 4])?;
                continue;
            };

            let item = unsafe { &*item_ptr };
            let current_pos = dim_bufman.cursor_position(cursor)?;
            let item_offset = item.serialize(
                dim_bufman,
                data_bufmans,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            dim_bufman.seek_with_cursor(cursor, current_pos)?;

            dim_bufman.update_u32_with_cursor(cursor, item_offset)?;
        }

        Ok(start as u32)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        data_file_parts: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;

        let placeholder_offset = file_offset.0 as u64;
        let array = Self::new();

        for i in 0..N {
            dim_bufman.seek_with_cursor(cursor, placeholder_offset + (i as u64 * 4))?;
            let offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                continue;
            }
            let item = Box::into_raw(Box::new(InvertedIndexSerialize::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(offset),
                data_file_idx,
                data_file_parts,
                cache,
            )?));
            array.insert(i, item);
        }

        dim_bufman.close_cursor(cursor)?;
        Ok(array)
    }
}
