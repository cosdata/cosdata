use std::{cell::UnsafeCell, sync::RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    tree_map::UnsafeVersionedItem,
    types::FileOffset,
    versioning::Hash,
};

use super::SimpleSerialize;

impl<T: SimpleSerialize> SimpleSerialize for UnsafeVersionedItem<T> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let next = unsafe { &*self.next.get() };
        let next_offset = if let Some(next) = next {
            next.serialize(bufman, cursor)?
        } else {
            u32::MAX
        };
        let offset_read_guard = self.serialized_at.read().map_err(|_| BufIoError::Locking)?;
        if let Some(FileOffset(offset)) = *offset_read_guard {
            bufman.seek_with_cursor(cursor, offset as u64 + 8)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            return Ok(offset);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self
            .serialized_at
            .write()
            .map_err(|_| BufIoError::Locking)?;
        if let Some(FileOffset(offset)) = *offset_write_guard {
            bufman.seek_with_cursor(cursor, offset as u64 + 8)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            return Ok(offset);
        }

        let mut buf = Vec::with_capacity(12);

        let value_offset = self.value.serialize(bufman, cursor)?;

        buf.extend(self.version.to_le_bytes());
        buf.extend(value_offset.to_le_bytes());
        buf.extend(next_offset.to_le_bytes());

        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;

        *offset_write_guard = Some(FileOffset(offset));

        Ok(offset)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let version = Hash::from(bufman.read_u32_with_cursor(cursor)?);
        let value_offset = bufman.read_u32_with_cursor(cursor)?;
        let next_offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        let value = T::deserialize(bufman, FileOffset(value_offset))?;
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
            value,
            next: UnsafeCell::new(next),
        })
    }
}
