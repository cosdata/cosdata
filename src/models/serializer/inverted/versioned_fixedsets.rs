use std::sync::RwLock;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    fixedset::{PerformantFixedSet, VersionedInvertedFixedSetIndex, INVERTED_FIXED_SET_INDEX_SIZE},
    types::FileOffset,
    versioning::Hash,
};

use super::InvertedIndexSerialize;

impl InvertedIndexSerialize for VersionedInvertedFixedSetIndex {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let buf = {
            let mut buf = Vec::with_capacity(
                9 + (self.exclusives.len() * INVERTED_FIXED_SET_INDEX_SIZE * 4)
                    * (2 + self.bits.len()),
            );
            let next = self.next.read().map_err(|_| BufIoError::Locking)?;
            let next_offset = if let Some(next) = &*next {
                next.serialize(dim_bufman, data_bufmans, data_file_idx, cursor)?
            } else {
                u32::MAX
            };
            buf.extend(next_offset.to_le_bytes());
            buf.extend(self.current_hash.to_le_bytes());
            buf.push(self.bits.len() as u8);
            for exclusive in &self.exclusives {
                for el in &exclusive.read().map_err(|_| BufIoError::Locking)?.buckets {
                    buf.extend(el.to_le_bytes());
                }
            }
            for bit in &self.bits {
                for el in &bit.read().map_err(|_| BufIoError::Locking)?.buckets {
                    buf.extend(el.to_le_bytes());
                }
            }
            buf
        };
        let data_bufman = data_bufmans.get(data_file_idx)?;
        let data_cursor = data_bufman.open_cursor()?;
        let data_offset =
            if let Some(offset) = *self.serialized_at.read().map_err(|_| BufIoError::Locking)? {
                data_bufman.seek_with_cursor(data_cursor, offset.0 as u64)?;
                data_bufman.update_with_cursor(data_cursor, &buf)?;
                offset.0
            } else {
                let mut serialized_at = self
                    .serialized_at
                    .write()
                    .map_err(|_| BufIoError::Locking)?;
                if let Some(offset) = *serialized_at {
                    data_bufman.seek_with_cursor(data_cursor, offset.0 as u64)?;
                    data_bufman.update_with_cursor(data_cursor, &buf)?;
                    return Ok(offset.0);
                } else {
                    let offset = data_bufman.write_to_end_of_file(data_cursor, &buf)? as u32;
                    *serialized_at = Some(FileOffset(offset));
                    offset
                }
            };
        data_bufman.close_cursor(data_cursor)?;
        let dim_offset = dim_bufman.cursor_position(cursor)? as u32;
        dim_bufman.update_u32_with_cursor(cursor, data_offset)?;
        Ok(dim_offset)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let bufman = data_bufmans.get(data_file_idx)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let next_offset = bufman.read_u32_with_cursor(cursor)?;
        let current_hash = Hash::from(bufman.read_u32_with_cursor(cursor)?);
        let quantization_bits = bufman.read_u8_with_cursor(cursor)?;
        let quantization_value = 1u32 << quantization_bits;
        let mut exclusives = Vec::with_capacity(quantization_value as usize);
        let mut bits = Vec::with_capacity(quantization_bits as usize);
        for _ in 0..quantization_value {
            let mut buckets = Vec::with_capacity(INVERTED_FIXED_SET_INDEX_SIZE);
            for _ in 0..INVERTED_FIXED_SET_INDEX_SIZE {
                let bucket = bufman.read_u64_with_cursor(cursor)?;
                buckets.push(bucket);
            }
            exclusives.push(RwLock::new(PerformantFixedSet { buckets }));
        }
        for _ in 0..quantization_bits {
            let len = (quantization_value >> 1) as usize * INVERTED_FIXED_SET_INDEX_SIZE;
            let mut buckets = Vec::with_capacity(len);
            for _ in 0..len {
                let bucket = bufman.read_u64_with_cursor(cursor)?;
                buckets.push(bucket);
            }
            bits.push(RwLock::new(PerformantFixedSet { buckets }));
        }
        bufman.close_cursor(cursor)?;
        let next = if next_offset == u32::MAX {
            None
        } else {
            Some(Box::new(VersionedInvertedFixedSetIndex::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(next_offset),
                data_file_idx,
                cache,
            )?))
        };

        Ok(Self {
            current_hash,
            serialized_at: RwLock::new(Some(file_offset)),
            exclusives,
            bits,
            next: RwLock::new(next),
        })
    }
}
