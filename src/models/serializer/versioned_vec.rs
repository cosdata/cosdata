use std::{cell::UnsafeCell, sync::RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    inverted_index_idf::UnsafeVersionedVec,
    types::FileOffset,
    versioning::Hash,
};

use super::SimpleSerialize;

impl SimpleSerialize for UnsafeVersionedVec<(u32, f32)> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let next = unsafe { &*self.next.get() };
        let next_offset = if let Some(next) = next {
            next.serialize(bufman, cursor)?
        } else {
            u32::MAX
        };
        let offset_read_guard = self.serialized_at.read().map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_read_guard {
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            return Ok(offset.0);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self
            .serialized_at
            .write()
            .map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_write_guard {
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            return Ok(offset.0);
        }
        let list = unsafe { &*self.list.get() };
        let size = 8 * list.len() + 12;
        let mut buf = Vec::with_capacity(size);
        buf.extend(next_offset.to_le_bytes());
        buf.extend(self.version.to_le_bytes());
        buf.extend((list.len() as u32).to_le_bytes());

        for (doc_id, tf) in list {
            buf.extend(doc_id.to_le_bytes());
            buf.extend(tf.to_le_bytes());
        }

        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        *offset_write_guard = Some(FileOffset(offset));

        Ok(offset)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let next_offset = bufman.read_u32_with_cursor(cursor)?;
        let version = Hash::from(bufman.read_u32_with_cursor(cursor)?);
        let len = bufman.read_u32_with_cursor(cursor)? as usize;
        let mut list = Vec::with_capacity(len);

        for _ in 0..len {
            let doc_id = bufman.read_u32_with_cursor(cursor)?;
            let tf = bufman.read_f32_with_cursor(cursor)?;
            list.push((doc_id, tf));
        }

        let next = if next_offset == u32::MAX {
            None
        } else {
            Some(Box::new(Self::deserialize(
                bufman,
                FileOffset(next_offset),
            )?))
        };

        Ok(Self {
            serialized_at: RwLock::new(Some(offset)),
            version,
            list: UnsafeCell::new(list),
            next: UnsafeCell::new(next),
        })
    }
}
