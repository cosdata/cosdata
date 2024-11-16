use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    ptr,
    sync::Arc,
};

use arcshift::ArcShift;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::{FileIndex, SyncPersist},
    prob_lazy_load::lazy_item::ProbLazyItem,
    types::{BytesToRead, FileOffset, HNSWLevel, ProbNode, PropState},
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

// Size in bytes: HNSW level (1)
//              + prop location (8)
//              + parent placeholder (10)
//              + child placeholder (10)
//              + neighbors placeholder (4)
//              = 33 bytes
impl ProbSerialize for ProbNode {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(&version)?;

        let mut buf = Vec::with_capacity(33);

        // Serialize basic fields
        buf.push(self.hnsw_level.0);

        // Serialize prop
        let mut prop = self.prop.clone();
        let prop_state = prop.get();
        match &*prop_state {
            PropState::Ready(node_prop) => {
                buf.extend(node_prop.location.0 .0.to_le_bytes());
                buf.extend(node_prop.location.1 .0.to_le_bytes());
            }
            PropState::Pending((FileOffset(offset), BytesToRead(length))) => {
                buf.extend(offset.to_le_bytes());
                buf.extend(length.to_le_bytes());
            }
        }

        // parent + child + neighbor placeholders
        buf.extend(vec![u8::MAX; 24]);

        let (start_offset, _) = bufman.write_to_end_with_cursor(cursor, &buf)?;

        let parent_ptr = self.get_parent();
        let child_ptr = self.get_child();

        // Serialize parent if present
        let parent_offset = if !parent_ptr.is_null() {
            Some(parent_ptr.serialize(bufmans.clone(), version, cursor)?)
        } else {
            None
        };

        // Serialize child if present
        let child_offset = if !child_ptr.is_null() {
            Some(child_ptr.serialize(bufmans.clone(), version, cursor)?)
        } else {
            None
        };

        // Serialize neighbors
        let neighbors_offset =
            self.get_neighbors_raw()
                .serialize(bufmans.clone(), version, cursor)?;

        if let Some(offset) = parent_offset {
            bufman.seek_with_cursor(cursor, SeekFrom::Start(start_offset + 9))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            unsafe {
                bufman.write_u16_with_cursor(cursor, (*parent_ptr).get_current_version_number())?;
                bufman.write_u32_with_cursor(cursor, *(*parent_ptr).get_current_version())?;
            }
        }

        if let Some(offset) = child_offset {
            bufman.seek_with_cursor(cursor, SeekFrom::Start(start_offset + 19))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            unsafe {
                bufman.write_u16_with_cursor(cursor, (*child_ptr).get_current_version_number())?;
                bufman.write_u32_with_cursor(cursor, *(*child_ptr).get_current_version())?;
            }
        }

        bufman.seek_with_cursor(cursor, SeekFrom::Start(start_offset + 29))?;
        bufman.write_u32_with_cursor(cursor, neighbors_offset)?;

        Ok(u32::try_from(start_offset).unwrap())
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<ProbCache>,
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
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                // Read basic fields
                let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
                // Read prop
                let prop_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
                let prop_length = BytesToRead(bufman.read_u32_with_cursor(cursor)?);
                let prop = PropState::Pending((prop_offset, prop_length));

                let parent_offset = bufman.read_u32_with_cursor(cursor)?;
                let parent_version_number = bufman.read_u16_with_cursor(cursor)?;
                let parent_version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);

                let child_offset = bufman.read_u32_with_cursor(cursor)?;
                let child_version_number = bufman.read_u16_with_cursor(cursor)?;
                let child_version_id = Hash::from(bufman.read_u32_with_cursor(cursor)?);

                let neighbors_offset = bufman.read_u32_with_cursor(cursor)?;
                bufman.close_cursor(cursor)?;
                // Deserialize parent
                let parent = if parent_offset != u32::MAX {
                    <*mut ProbLazyItem<Self>>::deserialize(
                        bufmans.clone(),
                        FileIndex::Valid {
                            offset: FileOffset(parent_offset),
                            version_number: parent_version_number,
                            version_id: parent_version_id,
                        },
                        cache.clone(),
                        max_loads,
                        skipm,
                    )?
                } else {
                    ptr::null_mut()
                };
                // Deserialize child
                let child = if child_offset != u32::MAX {
                    <*mut ProbLazyItem<Self>>::deserialize(
                        bufmans.clone(),
                        FileIndex::Valid {
                            offset: FileOffset(child_offset),
                            version_number: child_version_number,
                            version_id: child_version_id,
                        },
                        cache.clone(),
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
                    cache.clone(),
                    max_loads,
                    skipm,
                )?;

                Ok(Self::new_with_neighbors(
                    hnsw_level,
                    ArcShift::new(prop),
                    neighbors,
                    parent,
                    child,
                ))
            }
        }
    }
}

impl UpdateSerialized for ProbNode {
    fn update_serialized(
        &self,
        bufmans: Arc<BufferManagerFactory>,
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
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64 + 29))?;
                let neighbors_offset = bufman.read_u32_with_cursor(cursor)?;
                let neighbors_file_index = FileIndex::Valid {
                    offset: FileOffset(neighbors_offset),
                    version_number,
                    version_id,
                };
                self.get_neighbors_raw()
                    .update_serialized(bufmans, neighbors_file_index)?;

                Ok(offset)
            }
        }
    }
}
