use std::sync::{Arc, RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    page::{Pagepool, VersionedPagepool},
    types::FileOffset,
    versioning::Hash,
};

use super::SimpleSerialize;

impl<const LEN: usize> SimpleSerialize for VersionedPagepool<LEN> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let pagepool_offset = self.pagepool.serialize(bufman, cursor)?;
        let next_offset = {
            let next = self.next.read().map_err(|_| BufIoError::Locking)?;
            if let Some(next) = &*next {
                next.serialize(bufman, cursor)?
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

    fn deserialize(bufman: &BufferManager, file_offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let current_version = Hash::from(bufman.read_u32_with_cursor(cursor)?);
        let pagepool_offset = bufman.read_u32_with_cursor(cursor)?;
        let next_offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        let pagepool = Arc::new(Pagepool::deserialize(bufman, FileOffset(pagepool_offset))?);
        let next = if next_offset == u32::MAX {
            None
        } else {
            Some(Self::deserialize(bufman, FileOffset(next_offset))?)
        };
        Ok(Self {
            current_version,
            serialized_at: Arc::new(RwLock::new(Some(file_offset))),
            pagepool,
            next: Arc::new(RwLock::new(next)),
        })
    }
}
