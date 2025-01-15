use super::CustomSerialize;
use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::NodeRegistry,
    lazy_load::{EagerLazyItemSet, FileIndex, LazyItemRef},
    types::{BytesToRead, FileOffset, HNSWLevel, MergedNode, PropState},
    versioning::Hash,
};
use arcshift::ArcShift;
use std::collections::HashSet;
use std::{
    io::{self, SeekFrom},
    sync::Arc,
};

impl CustomSerialize for MergedNode {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start_offset = bufman.cursor_position(cursor)? as u32;

        // Serialize basic fields
        bufman.write_u8_with_cursor(cursor, self.hnsw_level.0)?;

        // Serialize prop
        let mut prop = self.prop.clone();
        let prop_state = prop.get();
        match &*prop_state {
            PropState::Ready(node_prop) => {
                bufman.write_u32_with_cursor(cursor, node_prop.location.0 .0)?;
                bufman.write_u32_with_cursor(cursor, node_prop.location.1 .0)?;
            }
            PropState::Pending((FileOffset(offset), BytesToRead(length))) => {
                bufman.write_u32_with_cursor(cursor, *offset)?;
                bufman.write_u32_with_cursor(cursor, *length)?;
            }
        }

        // Create and write indicator byte
        let mut indicator: u8 = 0;
        let parent_present = self.parent.is_valid();
        let child_present = self.child.is_valid();
        if parent_present {
            indicator |= 0b00000001;
        }
        if child_present {
            indicator |= 0b00000010;
        }
        bufman.write_u8_with_cursor(cursor, indicator)?;

        // Write placeholders only for present parent and child
        let parent_placeholder = if parent_present {
            let pos = bufman.cursor_position(cursor)? as u32;
            bufman.write_u32_with_cursor(cursor, 0)?;
            bufman.write_u16_with_cursor(cursor, 0)?;
            bufman.write_u32_with_cursor(cursor, 0)?;
            Some(pos)
        } else {
            None
        };

        let child_placeholder = if child_present {
            let pos = bufman.cursor_position(cursor)? as u32;
            bufman.write_u32_with_cursor(cursor, 0)?;
            bufman.write_u16_with_cursor(cursor, 0)?;
            bufman.write_u32_with_cursor(cursor, 0)?;
            Some(pos)
        } else {
            None
        };

        // Write placeholders for neighbors and versions
        let neighbors_placeholder = bufman.cursor_position(cursor)? as u32;
        bufman.write_u32_with_cursor(cursor, u32::MAX)?;

        // Serialize parent if present
        let parent_offset = if parent_present {
            Some(self.parent.serialize(bufmans.clone(), version, cursor)?)
        } else {
            None
        };

        // Serialize child if present
        let child_offset = if child_present {
            Some(self.child.serialize(bufmans.clone(), version, cursor)?)
        } else {
            None
        };

        // Serialize neighbors
        let neighbors_offset = self.neighbors.serialize(bufmans.clone(), version, cursor)?;

        // Update placeholders
        let end_pos = bufman.cursor_position(cursor)?;

        if let (Some(placeholder), Some(offset)) = (parent_placeholder, parent_offset) {
            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder as u64))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            bufman.write_u16_with_cursor(cursor, self.parent.get_current_version_number())?;
            bufman.write_u32_with_cursor(cursor, *self.parent.get_current_version())?;
        }

        if let (Some(placeholder), Some(offset)) = (child_placeholder, child_offset) {
            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder as u64))?;
            bufman.write_u32_with_cursor(cursor, offset)?;
            bufman.write_u16_with_cursor(cursor, self.child.get_current_version_number())?;
            bufman.write_u32_with_cursor(cursor, *self.child.get_current_version())?;
        }

        bufman.seek_with_cursor(cursor, SeekFrom::Start(neighbors_placeholder as u64))?;
        bufman.write_u32_with_cursor(cursor, neighbors_offset)?;

        // Return to the end of the serialized data
        bufman.seek_with_cursor(cursor, SeekFrom::Start(end_pos))?;

        Ok(start_offset)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize MergedNode with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_number,
                version_id,
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                // Read basic fields
                let hnsw_level = HNSWLevel(bufman.read_u8_with_cursor(cursor)?);
                // Read prop
                let prop_offset = FileOffset(bufman.read_u32_with_cursor(cursor)?);
                let prop_length = BytesToRead(bufman.read_u32_with_cursor(cursor)?);
                let prop = PropState::Pending((prop_offset, prop_length));
                // Read indicator byte
                let indicator = bufman.read_u8_with_cursor(cursor)?;
                let parent_present = indicator & 0b00000001 != 0;
                let child_present = indicator & 0b00000010 != 0;
                // Read offsets
                let mut parent_offset_and_version = None;
                let mut child_offset_and_version = None;
                if parent_present {
                    parent_offset_and_version = Some((
                        bufman.read_u32_with_cursor(cursor)?,
                        bufman.read_u16_with_cursor(cursor)?,
                        bufman.read_u32_with_cursor(cursor)?.into(),
                    ));
                }
                if child_present {
                    child_offset_and_version = Some((
                        bufman.read_u32_with_cursor(cursor)?,
                        bufman.read_u16_with_cursor(cursor)?,
                        bufman.read_u32_with_cursor(cursor)?.into(),
                    ));
                }
                let neighbors_offset = bufman.read_u32_with_cursor(cursor)?;
                bufman.close_cursor(cursor)?;
                // Deserialize parent
                let parent =
                    if let Some((offset, version_number, version_id)) = parent_offset_and_version {
                        LazyItemRef::deserialize(
                            bufmans.clone(),
                            FileIndex::Valid {
                                offset: FileOffset(offset),
                                version_number,
                                version_id,
                            },
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?
                    } else {
                        LazyItemRef::new_invalid()
                    };
                // Deserialize child
                let child =
                    if let Some((offset, version_number, version_id)) = child_offset_and_version {
                        LazyItemRef::deserialize(
                            bufmans.clone(),
                            FileIndex::Valid {
                                offset: FileOffset(offset),
                                version_number,
                                version_id,
                            },
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?
                    } else {
                        LazyItemRef::new_invalid()
                    };
                // Deserialize neighbors
                let neighbors = EagerLazyItemSet::deserialize(
                    bufmans,
                    FileIndex::Valid {
                        offset: FileOffset(neighbors_offset),
                        version_number,
                        version_id,
                    },
                    cache.clone(),
                    max_loads,
                    skipm,
                )?;

                Ok(MergedNode {
                    hnsw_level,
                    prop: ArcShift::new(prop),
                    neighbors,
                    parent,
                    child,
                })
            }
        }
    }
}
