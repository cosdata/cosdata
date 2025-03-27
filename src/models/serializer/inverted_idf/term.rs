use std::sync::{
    atomic::{AtomicU32, Ordering},
    RwLock,
};

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexIDFCache,
    common::TSHashTable,
    inverted_index_idf::TermInfo,
    page::VersionedPagepool,
    serializer::SimpleSerialize,
    types::FileOffset,
};

use super::InvertedIndexIDFSerialize;

impl InvertedIndexIDFSerialize for TermInfo {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        quantization_bits: u8,
        data_file_idx: u8,
        _data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let mut serialized_at = self
            .serialized_at
            .write()
            .map_err(|_| BufIoError::Locking)?;

        let start = if let Some(offset) = *serialized_at {
            dim_bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            offset.0
        } else {
            let offset =
                offset_counter.fetch_add((1u32 << quantization_bits) * 4 + 4, Ordering::Relaxed);
            dim_bufman.seek_with_cursor(cursor, offset as u64)?;
            offset
        };

        let data_bufman = data_bufmans.get(data_file_idx)?;
        let data_cursor = data_bufman.open_cursor()?;
        dim_bufman.update_u32_with_cursor(cursor, self.documents_count.load(Ordering::Relaxed))?;
        for i in 0..=((1u32 << quantization_bits) - 1) as u8 {
            let offset = self
                .frequency_map
                .with_value(&i, |pool| pool.serialize(&data_bufman, data_cursor))
                .unwrap_or(Ok(u32::MAX))?;
            dim_bufman.update_u32_with_cursor(cursor, offset)?;
        }
        *serialized_at = Some(FileOffset(start));
        data_bufman.close_cursor(cursor)?;
        Ok(start)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        quantization_bits: u8,
        data_file_idx: u8,
        _data_file_parts: u8,
        _cache: &InvertedIndexIDFCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let documents_count = AtomicU32::new(dim_bufman.read_u32_with_cursor(cursor)?);

        let frequency_map = TSHashTable::new(16);
        let data_bufman = data_bufmans.get(data_file_idx)?;

        for i in 0..=((1u32 << quantization_bits) - 1) as u8 {
            let offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if offset == u32::MAX {
                continue;
            }
            let list = VersionedPagepool::deserialize(&data_bufman, FileOffset(offset))?;
            frequency_map.insert(i, list);
        }

        Ok(Self {
            serialized_at: RwLock::new(Some(file_offset)),
            frequency_map,
            sequence_idx: 0, // Handled by caller
            documents_count,
        })
    }
}
