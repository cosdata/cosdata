use super::CustomSerialize;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{EagerLazyItemSet, LazyItemMap, LazyItemRef},
    types::{Item, MergedNode, PropState},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{atomic::AtomicBool, Arc},
};

use crate::models::types::FileOffset;
use std::collections::HashSet;
impl CustomSerialize for MergedNode {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let start_offset = writer.stream_position()? as u32;

        // Serialize basic fields
        writer.write_u16::<LittleEndian>(self.version_id.0)?;
        writer.write_u8(self.hnsw_level.0)?;

        // Serialize prop
        let mut prop = self.prop.clone();
        let prop_state = prop.get();
        match &*prop_state {
            PropState::Ready(node_prop) => {
                if let Some((FileOffset(offset), BytesToRead(length))) = node_prop.location {
                    writer.write_u32::<LittleEndian>(offset)?;
                    writer.write_u32::<LittleEndian>(length)?;
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Ready PropState with no location",
                    ));
                }
            }
            PropState::Pending((FileOffset(offset), BytesToRead(length))) => {
                writer.write_u32::<LittleEndian>(*offset)?;
                writer.write_u32::<LittleEndian>(*length)?;
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
        writer.write_u32::<LittleEndian>(u32::MAX)?;
        let versions_placeholder = writer.stream_position()? as u32;
        writer.write_u32::<LittleEndian>(u32::MAX)?;

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

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        FileOffset(offset): FileOffset,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        // Read basic fields
        let version_id = VersionId(reader.read_u16::<LittleEndian>()?);
        let hnsw_level = HNSWLevel(reader.read_u8()?);

        // Read prop
        let prop_offset = FileOffset(reader.read_u32::<LittleEndian>()?);
        let prop_length = BytesToRead(reader.read_u32::<LittleEndian>()?);
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
            LazyItemRef::deserialize(reader, offset, cache.clone(), max_loads, skipm)?
        } else {
            LazyItemRef::new_invalid()
        };

        // Deserialize child
        let child = if let Some(offset) = child_offset {
            LazyItemRef::deserialize(reader, offset, cache.clone(), max_loads, skipm)?
        } else {
            LazyItemRef::new_invalid()
        };

        // Deserialize neighbors
        let neighbors = EagerLazyItemSet::deserialize(
            reader,
            neighbors_offset,
            cache.clone(),
            max_loads,
            skipm,
        )?;

        // Deserialize versions
        let versions =
            LazyItemMap::deserialize(reader, versions_offset, cache.clone(), max_loads, skipm)?;

        Ok(MergedNode {
            version_id,
            hnsw_level,
            prop: Item::new(prop),
            neighbors,
            parent,
            child,
            versions,
            persist_flag: Arc::new(AtomicBool::new(true)),
        })
    }
}
