use std::{collections::HashMap, marker::PhantomData, sync::RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    lazy_item::FileIndex,
    om_versioned_vec::{LazyOmVersionedVec, OmVersionedVec},
    serializer::InlineSerialize,
    types::FileOffset,
    versioned_vec::VersionedVec,
    versioning::VersionNumber,
};

use super::TreeMapSerialize;

impl<T> TreeMapSerialize for VersionedVec<T> {
    fn serialize(
        &self,
        _dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        _cursor: u64,
    ) -> Result<u32, BufIoError> {
        // Collect all nodes in the linked list
        let mut nodes = Vec::new();
        let mut current = Some(self);
        while let Some(node) = current {
            nodes.push(node);
            current = node.next.as_deref();
        }

        // Serialize in reverse order (newest first)
        let mut next_offset = u32::MAX;
        let mut next_version = u32::MAX;

        for node in nodes.iter().rev() {
            let offset_read_guard = node.serialized_at.read().map_err(|_| BufIoError::Locking)?;
            if let Some(offset) = *offset_read_guard {
                // This node was already serialized, just update its next pointer
                let bufman = data_bufmans.get(node.version)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, offset.0 as u64)?;
                bufman.update_u32_with_cursor(cursor, next_offset)?;
                bufman.update_u32_with_cursor(cursor, next_version)?;
                bufman.close_cursor(cursor)?;
                next_offset = offset.0;
                next_version = *node.version;
                continue;
            }
            drop(offset_read_guard);

            let mut offset_write_guard = node
                .serialized_at
                .write()
                .map_err(|_| BufIoError::Locking)?;

            if let Some(offset) = *offset_write_guard {
                // This node was already serialized by another thread, just update its next pointer
                let bufman = data_bufmans.get(node.version)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, offset.0 as u64)?;
                bufman.update_u32_with_cursor(cursor, next_offset)?;
                bufman.update_u32_with_cursor(cursor, next_version)?;
                bufman.close_cursor(cursor)?;
                next_offset = offset.0;
                next_version = *node.version;
                continue;
            }

            // This is a new node, serialize it completely
            let size = 8 * node.list.len() + 16;
            let mut buf = Vec::with_capacity(size);
            buf.extend(next_offset.to_le_bytes());
            buf.extend(next_version.to_le_bytes());
            buf.extend((*node.version).to_le_bytes());
            buf.extend((node.list.len() as u32).to_le_bytes());

            for el in &node.list {
                buf.extend(el.to_le_bytes());
            }

            let bufman = data_bufmans.get(node.version)?;
            let cursor = bufman.open_cursor()?;
            let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
            *offset_write_guard = Some(FileOffset(offset));

            next_offset = offset;
            next_version = *node.version;
        }

        Ok(next_offset)
    }

    fn deserialize(
        _dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
    ) -> Result<Self, BufIoError> {
        let mut version_data: HashMap<VersionNumber, (FileOffset, Vec<u64>, VersionNumber)> =
            HashMap::new();
        let mut current_offset = file_offset;
        let mut current_version = version;

        // Read all versions in reverse order (newest first)
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

        // Rebuild the linked list in the original order (oldest first)
        // We need to determine the correct order by following the version chain
        let mut versions = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut current_version = version;

        while !visited.contains(&current_version)
            && current_version != VersionNumber::from(u32::MAX)
        {
            visited.insert(current_version);
            versions.push(current_version);
            if let Some((_, _, next_version)) = version_data.get(&current_version) {
                current_version = *next_version;
            } else {
                break;
            }
        }

        let mut current = None;
        for version in versions.iter().rev() {
            if let Some((file_offset, list, _)) = version_data.get(version) {
                let node = VersionedVec {
                    serialized_at: RwLock::new(Some(*file_offset)),
                    version: *version,
                    list: list.clone(),
                    next: current,
                    _marker: PhantomData,
                };
                current = Some(Box::new(node));
            }
        }

        Ok(*current.unwrap())
    }
}

impl<T: InlineSerialize> OmVersionedVec<T> {
    pub fn serialize(
        &mut self,
        bufmans: &BufferManagerFactory<VersionNumber>,
        version: VersionNumber,
    ) -> Result<(u32, VersionNumber), BufIoError> {
        let (prev_offset, prev_version) = if let Some(prev) = &mut self.prev {
            prev.serialize(bufmans, version)?
        } else {
            (u32::MAX, VersionNumber::from(u32::MAX))
        };

        if self.version != version {
            return Ok((prev_offset, prev_version));
        }

        let bufman = bufmans.get(self.version)?;

        let mut buf = Vec::with_capacity(8 + self.list.len() * T::SIZE);
        buf.extend(prev_offset.to_le_bytes());
        buf.extend(prev_version.to_le_bytes());
        buf.extend(self.version.to_le_bytes());
        buf.extend((self.list.len() as u32).to_le_bytes());

        for el in &self.list {
            el.serialize(&mut buf, &bufman)?;
        }

        let cursor = bufman.open_cursor()?;
        let offset = bufman.write_to_end_of_file(cursor, &buf)?;
        bufman.close_cursor(cursor)?;

        Ok((offset as u32, self.version))
    }
}

impl<T: InlineSerialize> LazyOmVersionedVec<T> {
    pub fn serialize(
        &mut self,
        bufmans: &BufferManagerFactory<VersionNumber>,
        version: VersionNumber,
    ) -> Result<(u32, VersionNumber), BufIoError> {
        match self {
            Self::Vec(vec) => {
                let (offset, vec_version) = vec.serialize(bufmans, version)?;
                if vec.version == version {
                    *self = Self::FileIndex(FileIndex {
                        offset: FileOffset(offset),
                        file_id: vec_version,
                    });
                }

                Ok((offset, vec_version))
            }
            Self::FileIndex(index) => Ok((index.offset.0, index.file_id)),
        }
    }

    pub fn deserialize(offset: FileOffset, version: VersionNumber) -> Self {
        Self::FileIndex(FileIndex {
            offset,
            file_id: version,
        })
    }
}
