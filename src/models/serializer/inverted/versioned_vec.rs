#![allow(clippy::only_used_in_recursion)]

use std::{collections::HashMap, marker::PhantomData, sync::RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    types::FileOffset,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

use super::InvertedIndexSerialize;

impl<T> InvertedIndexSerialize for VersionedVec<T> {
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
        let size = 8 * self.list.len() + 16;
        let mut buf = Vec::with_capacity(size);
        buf.extend(next_offset.to_le_bytes());
        buf.extend(next_version.to_le_bytes());
        buf.extend(self.version.to_le_bytes());
        buf.extend((self.list.len() as u32).to_le_bytes());
        let bufman = data_bufmans.get(self.version)?;
        let cursor = bufman.open_cursor()?;

        for el in &self.list {
            buf.extend(el.to_le_bytes());
        }

        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        *offset_write_guard = Some(FileOffset(offset));

        Ok(offset)
    }

    fn deserialize(
        _dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
        _cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        let mut version_data: HashMap<VersionNumber, (FileOffset, Vec<u64>, VersionNumber)> =
            HashMap::new();
        let mut current_offset = file_offset;
        let mut current_version = version;

        // Read all versions into version_data
        while current_offset.0 != u32::MAX {
            let bufman = data_bufmans.get(current_version)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, current_offset.0 as u64)?;
            let next_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
            let next_version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
            let version = VersionNumber::from(bufman.read_u32_with_cursor(cursor)?);
            let len = bufman.read_u32_with_cursor(cursor)? as usize;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(bufman.read_u64_with_cursor(cursor)?);
            }
            version_data.insert(version, (current_offset, list, next_version));
            current_offset = next_offset;
            current_version = next_version;
            bufman.close_cursor(cursor)?;
        }

        // Collect delete operations
        let mut deletes = Vec::new();
        for (_, list, _) in version_data.values() {
            for &item in list {
                if (item & (1 << 63)) != 0 {
                    let target_version = VersionNumber::from(((item >> 32) & 0x7FFFFFFF) as u32);
                    let target_index = (item & 0xFFFFFFFF) as usize;
                    deletes.push((target_version, target_index));
                }
            }
        }

        // Apply delete operations
        for (target_version, target_index) in deletes {
            if let Some((_, target_list, _)) = version_data.get_mut(&target_version) {
                if target_index < target_list.len() {
                    target_list[target_index] = u64::MAX;
                }
            }
        }

        // Build the VersionedVec chain
        let mut versions = Vec::new();
        let mut current_version = version;
        while let Some((_, _, next_version)) = version_data.get(&current_version) {
            versions.push(current_version);
            if *next_version == VersionNumber::from(u32::MAX) {
                break;
            }
            current_version = *next_version;
        }

        let mut current = None;
        for version in versions.iter().rev() {
            let (file_offset, list, _) = version_data.remove(version).unwrap();
            let node = VersionedVec {
                serialized_at: RwLock::new(Some(file_offset)),
                version: *version,
                list,
                next: current,
                _marker: PhantomData,
            };
            current = Some(Box::new(node));
        }

        Ok(*current.unwrap())
    }
}
