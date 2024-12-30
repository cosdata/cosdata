use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    ptr,
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::{FileIndex, SyncPersist},
    prob_lazy_load::lazy_item_array::ProbLazyItemArray,
    prob_node::{ProbNode, SharedNode},
    types::{BytesToRead, FileOffset, HNSWLevel},
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

impl ProbSerialize for ProbNode {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)?;

        // Serialize basic fields
        bufman.write_u8_with_cursor(cursor, self.hnsw_level.0)?;

        // Serialize prop
        let (FileOffset(offset), BytesToRead(length)) = &self.prop.location;
        bufman.write_u32_with_cursor(cursor, *offset)?;
        bufman.write_u32_with_cursor(cursor, *length)?;

        // 10 bytes for parent offset + 10 bytes for child offset + 4 bytes for neighbors offset + 4 bytes for versions
        bufman.write_with_cursor(cursor, &[u8::MAX; 28])?;

        let parent_ptr = self.get_parent();

        // Serialize parent if present
        let parent_file_index = if let Some(parent) = unsafe { parent_ptr.as_ref() } {
            Some((
                parent_ptr.serialize(bufmans, version, cursor)?,
                parent.get_current_version(),
                parent.get_current_version_number(),
            ))
        } else {
            None
        };

        let child_ptr = self.get_child();

        // Serialize child if present
        let child_file_index = if let Some(child) = unsafe { child_ptr.as_ref() } {
            Some((
                child_ptr.serialize(bufmans, version, cursor)?,
                child.get_current_version(),
                child.get_current_version_number(),
            ))
        } else {
            None
        };

        // Serialize neighbors
        let neighbors_offset = self
            .get_neighbors_raw()
            .serialize(bufmans, version, cursor)?;

        let versions_offset = self.versions.serialize(bufmans, version, cursor)?;

        let end_offset = bufman.cursor_position(cursor)?;

        if let Some((offset, version_id, version_number)) = parent_file_index {
            bufman.seek_with_cursor(cursor, SeekFrom::Start(start_offset + 9))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            bufman.write_u16_with_cursor(cursor, version_number)?;
            bufman.write_u32_with_cursor(cursor, *version_id)?;
        }

        if let Some((offset, version_id, version_number)) = child_file_index {
            bufman.seek_with_cursor(cursor, SeekFrom::Start(start_offset + 19))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            bufman.write_u16_with_cursor(cursor, version_number)?;
            bufman.write_u32_with_cursor(cursor, *version_id)?;
        }

        bufman.seek_with_cursor(cursor, SeekFrom::Start(start_offset + 29))?;
        bufman.write_u32_with_cursor(cursor, neighbors_offset)?;
        bufman.write_u32_with_cursor(cursor, versions_offset)?;

        bufman.seek_with_cursor(cursor, SeekFrom::Start(end_offset))?;

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize ProbNode with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                version_id,
                version_number,
                offset: FileOffset(offset),
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                // Read basic fields
                let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
                // Read prop
                let prop_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
                let prop_length = BytesToRead(bufman.read_u32_with_cursor(cursor)?);
                let prop = cache.get_prop(prop_offset, prop_length)?;

                let parent_offset = bufman.read_u32_with_cursor(cursor)?;
                let parent_version_number = bufman.read_u16_with_cursor(cursor)?;
                let parent_version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);

                let child_offset = bufman.read_u32_with_cursor(cursor)?;
                let child_version_number = bufman.read_u16_with_cursor(cursor)?;
                let child_version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);

                let neighbors_offset = bufman.read_u32_with_cursor(cursor)?;
                let versions_offset = bufman.read_u32_with_cursor(cursor)?;
                bufman.close_cursor(cursor)?;
                // Deserialize parent
                let parent = if parent_offset != u32::MAX {
                    SharedNode::deserialize(
                        bufmans,
                        FileIndex::Valid {
                            offset: FileOffset(parent_offset),
                            version_number: parent_version_number,
                            version_id: parent_version_id,
                        },
                        cache,
                        max_loads,
                        skipm,
                    )?
                } else {
                    ptr::null_mut()
                };
                // Deserialize child
                let child = if child_offset != u32::MAX {
                    SharedNode::deserialize(
                        bufmans,
                        FileIndex::Valid {
                            offset: FileOffset(child_offset),
                            version_number: child_version_number,
                            version_id: child_version_id,
                        },
                        cache,
                        max_loads,
                        skipm,
                    )?
                } else {
                    ptr::null_mut()
                };

                let neighbors_file_index = FileIndex::Valid {
                    offset: FileOffset(neighbors_offset),
                    version_number,
                    version_id,
                };
                // Deserialize neighbors
                let neighbors = ProbSerialize::deserialize(
                    bufmans,
                    neighbors_file_index,
                    cache,
                    max_loads,
                    skipm,
                )?;

                let versions_file_index = FileIndex::Valid {
                    offset: FileOffset(versions_offset),
                    version_number,
                    version_id,
                };

                let versions = ProbLazyItemArray::deserialize(
                    bufmans,
                    versions_file_index,
                    cache,
                    max_loads,
                    skipm,
                )?;

                Ok(Self::new_with_neighbors_and_versions(
                    hnsw_level, prop, neighbors, parent, child, versions,
                ))
            }
        }
    }
}

impl UpdateSerialized for ProbNode {
    fn update_serialized(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
    ) -> Result<u32, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize ProbNode with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                version_id,
                version_number,
                offset: FileOffset(offset),
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64 + 29))?;
                let neighbors_offset = bufman.read_u32_with_cursor(cursor)?;
                let versions_offset = bufman.read_u32_with_cursor(cursor)?;
                let neighbors_file_index = FileIndex::Valid {
                    offset: FileOffset(neighbors_offset),
                    version_number,
                    version_id,
                };
                let versions_file_index = FileIndex::Valid {
                    offset: FileOffset(versions_offset),
                    version_number,
                    version_id,
                };
                self.get_neighbors_raw()
                    .update_serialized(bufmans, neighbors_file_index)?;
                self.versions
                    .update_serialized(bufmans, versions_file_index)?;

                Ok(offset)
            }
        }
    }
}
