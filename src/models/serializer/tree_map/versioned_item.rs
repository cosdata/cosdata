#![allow(clippy::only_used_in_recursion)]

use parking_lot::RwLock;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    serializer::SimpleSerialize,
    tree_map::VersionedItem,
    types::FileOffset,
    versioning::VersionNumber,
};

use super::TreeMapSerialize;

impl<T: SimpleSerialize> TreeMapSerialize for VersionedItem<T> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let (next_offset, next_version) = if let Some(next) = &self.next {
            (
                next.serialize(dim_bufman, data_bufmans, cursor)?,
                *next.version,
            )
        } else {
            (u32::MAX, u32::MAX)
        };
        let offset_read_guard = self.serialized_at.read();
        if let Some(FileOffset(offset)) = *offset_read_guard {
            let bufman = data_bufmans.get(self.version)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, offset as u64)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            bufman.update_u32_with_cursor(cursor, next_version)?;
            bufman.close_cursor(cursor)?;
            return Ok(offset);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.serialized_at.write();
        if let Some(FileOffset(offset)) = *offset_write_guard {
            let bufman = data_bufmans.get(self.version)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, offset as u64)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            bufman.update_u32_with_cursor(cursor, next_version)?;
            bufman.close_cursor(cursor)?;
            return Ok(offset);
        }

        let mut buf = Vec::with_capacity(16);
        let bufman = data_bufmans.get(self.version)?;
        let cursor = bufman.open_cursor()?;

        let value_offset = if let Some(value) = &self.value {
            value.serialize(&bufman, cursor)?
        } else {
            u32::MAX
        };

        buf.extend(next_offset.to_le_bytes());
        buf.extend(next_version.to_le_bytes());
        buf.extend(self.version.to_le_bytes());
        buf.extend(value_offset.to_le_bytes());

        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        bufman.close_cursor(cursor)?;

        *offset_write_guard = Some(FileOffset(offset));

        Ok(offset)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        offset: FileOffset,
        version: VersionNumber,
    ) -> Result<Self, BufIoError> {
        let bufman = data_bufmans.get(version)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let next_offset = bufman.read_u32_with_cursor(cursor)?;
        let next_version = bufman.read_u32_with_cursor(cursor)?;
        let version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
        let value_offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        let value = if value_offset == u32::MAX {
            None
        } else {
            Some(T::deserialize(&bufman, FileOffset(value_offset))?)
        };
        let next = if next_offset == u32::MAX {
            None
        } else {
            Some(Box::new(Self::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(next_offset),
                VersionNumber::from(next_version),
            )?))
        };

        Ok(Self {
            serialized_at: RwLock::new(Some(offset)),
            version,
            value,
            next,
        })
    }
}
