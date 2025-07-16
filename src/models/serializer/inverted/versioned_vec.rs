#![allow(clippy::only_used_in_recursion)]

use std::sync::RwLock;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    serializer::SimpleSerialize,
    types::FileOffset,
    versioned_vec::{VersionedVec, VersionedVecItem},
    versioning::VersionNumber,
};

use super::InvertedIndexSerialize;

impl<T> InvertedIndexSerialize for VersionedVec<T>
where
    T: SimpleSerialize + VersionedVecItem,
    <T as VersionedVecItem>::Id: SimpleSerialize,
{
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let (next_offset, next_version) = if let Some(next) = &self.next {
            (
                InvertedIndexSerialize::serialize(next.as_ref(), dim_bufman, data_bufmans, cursor)?,
                *next.version,
            )
        } else {
            (u32::MAX, u32::MAX)
        };
        let offset_read_guard = self.serialized_at.read().map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_read_guard {
            let bufman = data_bufmans.get(self.version)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            bufman.update_u32_with_cursor(cursor, next_version)?;
            bufman.close_cursor(cursor)?;
            return Ok(offset.0);
        }
        drop(offset_read_guard);
        let mut offset_write_guard = self
            .serialized_at
            .write()
            .map_err(|_| BufIoError::Locking)?;
        if let Some(offset) = *offset_write_guard {
            let bufman = data_bufmans.get(self.version)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, offset.0 as u64)?;
            bufman.update_u32_with_cursor(cursor, next_offset)?;
            bufman.update_u32_with_cursor(cursor, next_version)?;
            bufman.close_cursor(cursor)?;
            return Ok(offset.0);
        }
        let size = 4 * self.list.len() + 4 * self.deleted.len() + 20;
        let mut buf = Vec::with_capacity(size);
        buf.extend(next_offset.to_be_bytes());
        buf.extend(next_version.to_be_bytes());
        buf.extend(self.version.to_le_bytes());
        buf.extend((self.list.len() as u32).to_le_bytes());
        buf.extend((self.deleted.len() as u32).to_le_bytes());
        let bufman = data_bufmans.get(self.version)?;
        let cursor = bufman.open_cursor()?;

        for el in &self.list {
            let serialized_offset = el.serialize(&bufman, cursor)?;
            buf.extend(serialized_offset.to_le_bytes());
        }

        for el in &self.deleted {
            let serialized_offset = el.serialize(&bufman, cursor)?;
            buf.extend(serialized_offset.to_le_bytes());
        }

        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        *offset_write_guard = Some(FileOffset(offset));

        Ok(offset)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let bufman = data_bufmans.get(version)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let next_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
        let next_version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
        let version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
        let len = bufman.read_u32_with_cursor(cursor)? as usize;
        let deleted_len = bufman.read_u32_with_cursor(cursor)? as usize;
        let mut list = Vec::with_capacity(len);
        let mut deleted = Vec::with_capacity(deleted_len);

        for _ in 0..len {
            let el_offset = bufman.read_u32_with_cursor(cursor)?;
            let el = T::deserialize(&bufman, FileOffset(el_offset))?;
            list.push(el);
        }

        for _ in 0..deleted_len {
            let el_offset = bufman.read_u32_with_cursor(cursor)?;
            let el = <T as VersionedVecItem>::Id::deserialize(&bufman, FileOffset(el_offset))?;
            deleted.push(el);
        }

        let next = if next_offset.0 == u32::MAX {
            None
        } else {
            Some(Box::new(InvertedIndexSerialize::deserialize(
                dim_bufman,
                data_bufmans,
                next_offset,
                next_version,
                cache,
            )?))
        };

        Ok(Self {
            serialized_at: RwLock::new(Some(file_offset)),
            version,
            list,
            deleted,
            next,
        })
    }
}
