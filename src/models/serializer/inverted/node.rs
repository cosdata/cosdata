use std::sync::atomic::{AtomicBool, Ordering};

use crate::models::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    inverted_index::{InvertedIndexNode, InvertedIndexNodeData},
    lazy_item::LazyItem,
    types::FileOffset,
    versioning::VersionNumber,
};

use super::InvertedIndexSerialize;

// @SERIALIZED_SIZE:
//
//   qb = quantization bits (4, 5, 6)
//   qv = quantization value = 2^qb (16, 32, 64)
//
//   4 byte for dim index +                           | 4
//   1 byte for implicit flag & quantization          | 5
//   quantization value *                             |
//   4 + 4 bytes for versioned vec offset & version + | qv * 8 + 5
//   16 * 4 bytes for dimension offsets               | qv * 8 + 69
impl InvertedIndexSerialize for InvertedIndexNode {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
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
            self.data.serialize(dim_bufman, data_bufmans, cursor)?;
            self.children.serialize(dim_bufman, data_bufmans, cursor)?;
        } else if self.is_dirty.swap(false, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64 + 5)?;
            self.data.serialize(dim_bufman, data_bufmans, cursor)?;
            self.children.serialize(dim_bufman, data_bufmans, cursor)?;
        } else {
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + 5 + (1u64 << (self.quantization_bits + 2)),
            )?;
            self.children.serialize(dim_bufman, data_bufmans, cursor)?;
        };
        Ok(self.file_offset.0)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
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
        let data = <*mut LazyItem<InvertedIndexNodeData, ()>>::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 5),
            version,
            cache,
        )?;
        let children = AtomicArray::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 5 + qv * 8),
            version,
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
        })
    }
}
