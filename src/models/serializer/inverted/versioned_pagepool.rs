use std::sync::{Arc, RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    page::{Pagepool, VersionedPagepool},
    types::FileOffset,
    versioning::Hash,
};

use super::InvertedIndexSerialize;

impl<const LEN: usize> InvertedIndexSerialize for VersionedPagepool<LEN> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = data_bufmans.get(data_file_idx)?;
        let pagepool_offset = self.pagepool.serialize(
            dim_bufman,
            data_bufmans,
            data_file_idx,
            data_file_parts,
            cursor,
        )?;
        let next_offset = {
            let next = self.next.read().map_err(|_| BufIoError::Locking)?;
            if let Some(next) = &*next {
                next.serialize(
                    dim_bufman,
                    data_bufmans,
                    data_file_idx,
                    data_file_parts,
                    cursor,
                )?
            } else {
                u32::MAX
            }
        };
        if let Some(offset) = *self.serialized_at.read().map_err(|_| BufIoError::Locking)? {
            bufman.seek_with_cursor(cursor, offset.0 as u64 + 4)?;
            bufman.update_u32_with_cursor(cursor, pagepool_offset)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            return Ok(offset.0);
        }
        let mut serialized_at = self
            .serialized_at
            .write()
            .map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *serialized_at {
            bufman.seek_with_cursor(cursor, offset.0 as u64 + 4)?;
            bufman.update_u32_with_cursor(cursor, pagepool_offset)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            return Ok(offset.0);
        }
        let mut buf = Vec::with_capacity(12);
        buf.extend(self.current_version.to_le_bytes());
        buf.extend(pagepool_offset.to_le_bytes());
        buf.extend(next_offset.to_le_bytes());
        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        *serialized_at = Some(FileOffset(offset));
        Ok(offset)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        data_file_parts: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let bufman = data_bufmans.get(data_file_idx)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let current_version = Hash::from(bufman.read_u32_with_cursor(cursor)?);
        let pagepool_offset = bufman.read_u32_with_cursor(cursor)?;
        let next_offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        let pagepool = Pagepool::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(pagepool_offset),
            data_file_idx,
            data_file_parts,
            cache,
        )?;
        let next = if next_offset == u32::MAX {
            None
        } else {
            Some(Self::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(next_offset),
                data_file_idx,
                data_file_parts,
                cache,
            )?)
        };
        Ok(Self {
            current_version,
            serialized_at: Arc::new(RwLock::new(Some(file_offset))),
            pagepool,
            next: Arc::new(RwLock::new(next)),
        })
    }
}
