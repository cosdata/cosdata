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
        let offset_read_guard = self.serialized_at.read();
        if let Some(FileOffset(offset)) = *offset_read_guard {
            return Ok(offset);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self.serialized_at.write();
        if let Some(FileOffset(offset)) = *offset_write_guard {
            return Ok(offset);
        }
        let (prev_offset, prev_version) = if let Some(prev) = &self.prev {
            (
                prev.serialize(dim_bufman, data_bufmans, cursor)?,
                *prev.version,
            )
        } else {
            (u32::MAX, u32::MAX)
        };

        let mut buf = Vec::with_capacity(16);
        let bufman = data_bufmans.get(self.version)?;
        let cursor = bufman.open_cursor()?;

        let value_offset = if let Some(value) = &self.value {
            value.serialize(&bufman, cursor)?
        } else {
            u32::MAX
        };

        buf.extend(prev_offset.to_le_bytes());
        buf.extend(prev_version.to_le_bytes());
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
        let prev_offset = bufman.read_u32_with_cursor(cursor)?;
        let prev_version = bufman.read_u32_with_cursor(cursor)?;
        let version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
        let value_offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        let value = if value_offset == u32::MAX {
            None
        } else {
            Some(T::deserialize(&bufman, FileOffset(value_offset))?)
        };
        let prev = if prev_offset == u32::MAX {
            None
        } else {
            Some(Box::new(Self::deserialize(
                dim_bufman,
                data_bufmans,
                FileOffset(prev_offset),
                VersionNumber::from(prev_version),
            )?))
        };

        Ok(Self {
            serialized_at: RwLock::new(Some(offset)),
            version,
            value,
            prev,
        })
    }
}
