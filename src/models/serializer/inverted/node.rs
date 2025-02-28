use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    models::{
        atomic_array::AtomicArray,
        buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
        cache_loader::InvertedIndexCache,
        fixedset::VersionedInvertedFixedSetIndex,
        prob_lazy_load::lazy_item::ProbLazyItem,
        types::FileOffset,
    },
    storage::inverted_index_sparse_ann_basic::{
        InvertedIndexSparseAnnNodeBasicTSHashmap, InvertedIndexSparseAnnNodeBasicTSHashmapData,
    },
};

use super::{InvertedIndexSerialize, DATA_FILE_PARTS};

// @SERIALIZED_SIZE:
//
//   qb = quantization bits (4, 5, 6)
//   qv = quantization value = 2^qb (16, 32, 64)
//
//   4 byte for dim index +                          | 4
//   1 byte for implicit flag & quantization         | 5
//   quantization value *                            |
//   4 bytes for pagepool offset +                   | qv * 4 + 5
//   16 * 4 bytes for dimension offsets +            | qv * 4 + 69
//   4 bytes of sets offset                          | qv * 4 + 73
impl InvertedIndexSerialize for InvertedIndexSparseAnnNodeBasicTSHashmap {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        _: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        if !self.is_serialized.swap(true, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64)?;
            dim_bufman.update_u32_with_cursor(cursor, self.dim_index)?;
            let mut quantization_and_implicit = self.quantization_bits;
            if self.implicit {
                quantization_and_implicit |= 1u8 << 7;
            }
            dim_bufman.update_u8_with_cursor(cursor, quantization_and_implicit)?;
            let data_file_idx = (self.dim_index % DATA_FILE_PARTS) as u8;
            self.data
                .serialize(dim_bufman, data_bufmans, data_file_idx, cursor)?;
            self.children
                .serialize(dim_bufman, data_bufmans, data_file_idx, cursor)?;
            self.fixed_sets
                .serialize(dim_bufman, data_bufmans, data_file_idx, cursor)?;
        } else if self.is_dirty.swap(false, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64 + 5)?;
            let data_file_idx = (self.dim_index % DATA_FILE_PARTS) as u8;
            self.data
                .serialize(dim_bufman, data_bufmans, data_file_idx, cursor)?;
            self.children
                .serialize(dim_bufman, data_bufmans, data_file_idx, cursor)?;
            self.fixed_sets
                .serialize(dim_bufman, data_bufmans, data_file_idx, cursor)?;
        };
        Ok(self.file_offset.0)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        _: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let dim_index = dim_bufman.read_u32_with_cursor(cursor)?;
        let quantization_and_implicit = dim_bufman.read_u8_with_cursor(cursor)?;
        let implicit = (quantization_and_implicit & (1u8 << 7)) != 0;
        let quantization_bits = (quantization_and_implicit << 1) >> 1;
        let qb = quantization_bits as u32;
        let qv = 1u32 << qb;
        let data_file_idx = (dim_index % DATA_FILE_PARTS) as u8;
        let data = <*mut ProbLazyItem<InvertedIndexSparseAnnNodeBasicTSHashmapData>>::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 5),
            data_file_idx,
            cache,
        )?;
        let children = AtomicArray::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 5 + qv * 4),
            data_file_idx,
            cache,
        )?;
        let fixed_sets = <*mut ProbLazyItem<VersionedInvertedFixedSetIndex>>::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 69 + qv * 4),
            data_file_idx,
            cache,
        )?;

        Ok(Self {
            is_serialized: AtomicBool::new(true),
            is_dirty: AtomicBool::new(false),
            dim_index,
            implicit,
            file_offset,
            quantization_bits,
            data,
            children,
            fixed_sets,
        })
    }
}
