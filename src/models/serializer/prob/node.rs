use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    ptr,
    sync::atomic::AtomicPtr,
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::{FileIndex, SyncPersist},
    prob_lazy_load::lazy_item_array::ProbLazyItemArray,
    prob_node::{ProbNode, SharedNode},
    types::{BytesToRead, FileOffset, HNSWLevel, MetricResult},
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

// @SERIALIZED_SIZE:
//   Properties:
//     1 byte for HNSW level +             | 1
//     8 bytes for prop offset & length    | 1 + 8 = 9
//
//   Links:
//     10 bytes for parent offset & version +           | 10
//     10 bytes for child offset & version +            | 20
//     2 bytes for neighbors length +                   | 22
//     neighbors length * 19 bytes for neighbor link +  | nb * 19 + 22
//     8 * 10 bytes for version link                    | nb * 19 + 102
//
//   Total = nb * 19 + 111 (where `nb` is the neighbors count)
impl ProbSerialize for ProbNode {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        level_0_bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
        direct: bool,
        _is_level_0: bool,
    ) -> Result<u32, BufIoError> {
        let is_level_0 = self.hnsw_level.0 == 0;
        assert_eq!(_is_level_0, is_level_0);
        let bufman = if is_level_0 {
            level_0_bufmans.get(version)?
        } else {
            bufmans.get(version)?
        };
        let start_offset = bufman.cursor_position(cursor)?;

        let neighbors = self.get_neighbors_raw();
        let size = Self::get_serialized_size(neighbors.len()) as u64;

        assert_eq!(start_offset % size, 0, "offset: {}", start_offset);

        // Serialize basic fields
        bufman.write_u8_with_cursor(cursor, self.hnsw_level.0)?;

        // Serialize prop
        let (FileOffset(offset), BytesToRead(length)) = &self.prop.location;
        bufman.write_u32_with_cursor(cursor, *offset)?;
        bufman.write_u32_with_cursor(cursor, *length)?;

        bufman.write_with_cursor(cursor, &vec![u8::MAX; neighbors.len() * 19 + 102])?;

        let parent_ptr = self.get_parent();

        // Serialize parent if present
        let parent_file_index = if let Some(parent) = unsafe { parent_ptr.as_ref() } {
            let (bufman, cursor, new) = if is_level_0 {
                let parent_version = parent.get_current_version();
                let bufman = bufmans.get(parent_version)?;
                let cursor = bufman.open_cursor()?;
                (bufman, cursor, true)
            } else {
                (bufman.clone(), cursor, false)
            };
            let parent_offset =
                parent_ptr.serialize(bufmans, level_0_bufmans, version, cursor, direct, false)?;
            if new {
                bufman.close_cursor(cursor)?;
            }
            Some((
                parent_offset,
                parent.get_current_version(),
                parent.get_current_version_number(),
            ))
        } else {
            None
        };

        let child_ptr = self.get_child();

        // Serialize child if present
        let child_file_index = if let Some(child) = unsafe { child_ptr.as_ref() } {
            let (bufman, cursor, new) = if self.hnsw_level.0 == 1 {
                let child_version = child.get_current_version();
                let bufman = level_0_bufmans.get(child_version)?;
                let cursor = bufman.open_cursor()?;
                (bufman, cursor, true)
            } else {
                (bufman.clone(), cursor, false)
            };
            let child_offset = child_ptr.serialize(
                bufmans,
                level_0_bufmans,
                version,
                cursor,
                direct,
                self.hnsw_level.0 == 1,
            )?;
            if new {
                bufman.close_cursor(cursor)?;
            }
            Some((
                child_offset,
                child.get_current_version(),
                child.get_current_version_number(),
            ))
        } else {
            None
        };

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
        neighbors.serialize(
            bufmans,
            level_0_bufmans,
            version,
            cursor,
            direct,
            is_level_0,
        )?;
        bufman.seek_with_cursor(
            cursor,
            SeekFrom::Start(start_offset + 31 + neighbors.len() as u64 * 19),
        )?;
        self.versions.serialize(
            bufmans,
            level_0_bufmans,
            version,
            cursor,
            direct,
            is_level_0,
        )?;

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        level_0_bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
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
                let bufman = if is_level_0 {
                    level_0_bufmans.get(version_id)?
                } else {
                    bufmans.get(version_id)?
                };
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                // Read basic fields
                let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
                if is_level_0 {
                    assert_eq!(hnsw_level.0, 0);
                } else {
                    assert_ne!(hnsw_level.0, 0);
                }
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
                bufman.close_cursor(cursor)?;
                // Deserialize parent
                let parent = if parent_offset != u32::MAX {
                    SharedNode::deserialize(
                        bufmans,
                        level_0_bufmans,
                        FileIndex::Valid {
                            offset: FileOffset(parent_offset),
                            version_number: parent_version_number,
                            version_id: parent_version_id,
                        },
                        cache,
                        max_loads,
                        skipm,
                        false,
                    )?
                } else {
                    ptr::null_mut()
                };
                // Deserialize child
                let child = if child_offset != u32::MAX {
                    SharedNode::deserialize(
                        bufmans,
                        level_0_bufmans,
                        FileIndex::Valid {
                            offset: FileOffset(child_offset),
                            version_number: child_version_number,
                            version_id: child_version_id,
                        },
                        cache,
                        max_loads,
                        skipm,
                        hnsw_level.0 == 1,
                    )?
                } else {
                    ptr::null_mut()
                };

                let neighbors_file_index = FileIndex::Valid {
                    offset: FileOffset(offset + 29),
                    version_number,
                    version_id,
                };

                let neighbors: Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> =
                    ProbSerialize::deserialize(
                        bufmans,
                        level_0_bufmans,
                        neighbors_file_index,
                        cache,
                        max_loads,
                        skipm,
                        is_level_0,
                    )?;

                let versions_file_index = FileIndex::Valid {
                    offset: FileOffset(offset + 31 + neighbors.len() as u32 * 19),
                    version_number,
                    version_id,
                };

                let versions = ProbLazyItemArray::deserialize(
                    bufmans,
                    level_0_bufmans,
                    versions_file_index,
                    cache,
                    max_loads,
                    skipm,
                    is_level_0,
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
        level_0_bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        FileOffset(offset): FileOffset,
        cursor: u64,
        _is_level_0: bool,
    ) -> Result<u32, BufIoError> {
        let is_level_0 = self.hnsw_level.0 == 0;
        assert_eq!(_is_level_0, is_level_0);
        let bufman = if is_level_0 {
            level_0_bufmans.get(version)?
        } else {
            bufmans.get(version)?
        };
        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64 + 29))?;
        self.get_neighbors_raw().serialize(
            bufmans,
            level_0_bufmans,
            version,
            cursor,
            true,
            is_level_0,
        )?;
        bufman.seek_with_cursor(
            cursor,
            SeekFrom::Start(offset as u64 + 31 + self.get_neighbors_raw().len() as u64 * 19),
        )?;
        self.versions
            .serialize(bufmans, level_0_bufmans, version, cursor, true, is_level_0)?;

        Ok(offset)
    }
}
