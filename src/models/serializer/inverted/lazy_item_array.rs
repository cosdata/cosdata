use std::collections::HashSet;

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::InvertedIndexCache,
        lazy_load::FileIndex,
        prob_lazy_load::lazy_item_array::ProbLazyItemArray,
        types::FileOffset,
        versioning::Hash,
    },
    storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasicTSHashmap,
};

use super::{InvertedIndexFileIndex, InvertedIndexSerialize};

impl<const N: usize> InvertedIndexSerialize<Hash>
    for ProbLazyItemArray<InvertedIndexSparseAnnNodeBasicTSHashmap, N>
{
    fn serialize(
        &self,
        dim_bufmans: &BufferManagerFactory<Hash>,
        _data_bufmans: &BufferManagerFactory<(Hash, u8)>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = dim_bufmans.get(version)?;
        let start = bufman.cursor_position(cursor)?;
        bufman.update_with_cursor(cursor, &vec![u8::MAX; 10 * N])?;
        bufman.seek_with_cursor(cursor, start)?;

        for i in 0..N {
            let Some(item_ptr) = self.get(i) else {
                bufman.update_with_cursor(cursor, &[u8::MAX; 10])?;
                continue;
            };

            let item = unsafe { &*item_ptr };

            let (offset, version_number, version_id) = match item.get_file_index() {
                FileIndex::Valid {
                    offset,
                    version_number,
                    version_id,
                } => (offset.0, version_number, version_id),
                _ => unreachable!(),
            };

            bufman.update_u32_with_cursor(cursor, offset)?;
            bufman.update_u16_with_cursor(cursor, version_number)?;
            bufman.update_u32_with_cursor(cursor, *version_id)?;
        }

        Ok(start as u32)
    }

    fn deserialize(
        dim_bufmans: &BufferManagerFactory<Hash>,
        data_bufmans: &BufferManagerFactory<(Hash, u8)>,
        file_index: InvertedIndexFileIndex<Hash>,
        cache: &InvertedIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let bufman = dim_bufmans.get(file_index.file_identifier)?;
        let cursor = bufman.open_cursor()?;

        let placeholder_offset = file_index.offset.0 as u64;
        let array = Self::new();

        for i in 0..N {
            bufman.seek_with_cursor(cursor, placeholder_offset + (i as u64 * 10))?;
            let offset = bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                continue;
            }
            let version_number = bufman.read_u16_with_cursor(cursor)?;
            let version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);
            let file_index = InvertedIndexFileIndex {
                offset: FileOffset(offset),
                version_number,
                file_identifier: version_id,
            };
            let item = InvertedIndexSerialize::deserialize(
                dim_bufmans,
                data_bufmans,
                file_index,
                cache,
                max_loads,
                skipm,
            )?;
            array.insert(i, item);
        }

        bufman.close_cursor(cursor)?;
        Ok(array)
    }
}
