use super::CustomSerialize;
use crate::models::{
    chunked_list::{LazyItem, LazyItemRef, LazyItems, SyncPersist},
    types::{MergedNode, PropState},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, RwLock},
};

impl CustomSerialize for MergedNode {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        if !self.needs_persistence() {
            return Ok(u32::MAX);
        }

        self.set_persistence(false);

        let start_offset = writer.stream_position()? as u32;

        // Serialize basic fields
        writer.write_u16::<LittleEndian>(self.version_id)?;
        writer.write_u8(self.hnsw_level)?;

        // Serialize prop
        let prop_state = self.prop.read().unwrap();
        match &*prop_state {
            PropState::Ready(node_prop) => {
                if let Some((offset, length)) = node_prop.location {
                    writer.write_u32::<LittleEndian>(offset)?;
                    writer.write_u32::<LittleEndian>(length)?;
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Ready PropState with no location",
                    ));
                }
            }
            PropState::Pending((offset, length)) => {
                writer.write_u32::<LittleEndian>(*offset)?;
                writer.write_u32::<LittleEndian>(*length)?;
            }
        }

        // Create and write indicator byte
        let mut indicator: u8 = 0;
        let parent_present = self.parent.is_some();
        let child_present = self.child.is_some();
        if parent_present {
            indicator |= 0b00000001;
        }
        if child_present {
            indicator |= 0b00000010;
        }
        writer.write_u8(indicator)?;

        // Write placeholders only for present parent and child
        let parent_placeholder = if parent_present {
            let pos = writer.stream_position()? as u32;
            writer.write_u32::<LittleEndian>(0)?;
            Some(pos)
        } else {
            None
        };

        let child_placeholder = if child_present {
            let pos = writer.stream_position()? as u32;
            writer.write_u32::<LittleEndian>(0)?;
            Some(pos)
        } else {
            None
        };

        // Write placeholders for neighbors and versions
        let neighbors_placeholder = writer.stream_position()? as u32;
        writer.write_u32::<LittleEndian>(0)?;
        let versions_placeholder = writer.stream_position()? as u32;
        writer.write_u32::<LittleEndian>(0)?;

        // Serialize parent if present
        let parent_offset = if parent_present {
            Some(self.parent.serialize(writer)?)
        } else {
            None
        };

        // Serialize child if present
        let child_offset = if child_present {
            Some(self.child.serialize(writer)?)
        } else {
            None
        };

        // Serialize neighbors
        let neighbors_offset = self.neighbors.serialize(writer)?;

        // Serialize versions
        let versions_offset = self.versions.serialize(writer)?;

        // Update placeholders
        let end_pos = writer.stream_position()?;

        if let (Some(placeholder), Some(offset)) = (parent_placeholder, parent_offset) {
            writer.seek(SeekFrom::Start(placeholder as u64))?;
            writer.write_u32::<LittleEndian>(offset)?;
        }

        if let (Some(placeholder), Some(offset)) = (child_placeholder, child_offset) {
            writer.seek(SeekFrom::Start(placeholder as u64))?;
            writer.write_u32::<LittleEndian>(offset)?;
        }

        writer.seek(SeekFrom::Start(neighbors_placeholder as u64))?;
        writer.write_u32::<LittleEndian>(neighbors_offset)?;
        writer.seek(SeekFrom::Start(versions_placeholder as u64))?;
        writer.write_u32::<LittleEndian>(versions_offset)?;

        // Return to the end of the serialized data
        writer.seek(SeekFrom::Start(end_pos))?;

        Ok(start_offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        // Read basic fields
        let version_id = reader.read_u16::<LittleEndian>()?;
        let hnsw_level = reader.read_u8()?;

        // Read prop
        let prop_offset = reader.read_u32::<LittleEndian>()?;
        let prop_length = reader.read_u32::<LittleEndian>()?;
        let prop = PropState::Pending((prop_offset, prop_length));

        // Read indicator byte
        let indicator = reader.read_u8()?;
        let parent_present = indicator & 0b00000001 != 0;
        let child_present = indicator & 0b00000010 != 0;

        // Read offsets
        let mut parent_offset = None;
        let mut child_offset = None;
        if parent_present {
            parent_offset = Some(reader.read_u32::<LittleEndian>()?);
        }
        if child_present {
            child_offset = Some(reader.read_u32::<LittleEndian>()?);
        }
        let neighbors_offset = reader.read_u32::<LittleEndian>()?;
        let versions_offset = reader.read_u32::<LittleEndian>()?;

        // Deserialize parent
        let parent = if let Some(offset) = parent_offset {
            Some(LazyItemRef::deserialize(reader, offset)?)
        } else {
            None
        };

        // Deserialize child
        let child = if let Some(offset) = child_offset {
            Some(LazyItemRef::deserialize(reader, offset)?)
        } else {
            None
        };

        // Deserialize neighbors
        let neighbors = LazyItems::deserialize(reader, neighbors_offset)?;

        // Deserialize versions
        let versions = LazyItems::deserialize(reader, versions_offset)?;

        Ok(MergedNode {
            version_id,
            hnsw_level,
            prop: Arc::new(RwLock::new(prop)),
            neighbors,
            parent,
            child,
            versions,
            persist_flag: Arc::new(RwLock::new(true)),
        })
    }
}
