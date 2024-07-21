use crate::models::file_persist::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::transmute;

pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized;
}

impl CustomSerialize for NodePersistRef {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        match self {
            NodePersistRef::Reference(node) => {
                let node_offset = node.serialize(writer)?;
                writer.seek(SeekFrom::Start(offset as u64))?;
                writer.write_u32::<LittleEndian>(node_offset)?;
            }
            NodePersistRef::DerefPending(ptr) => {
                writer.write_u32::<LittleEndian>(*ptr)?;
            }
            NodePersistRef::Invalid => {
                writer.write_u32::<LittleEndian>(u32::MAX - 1)?;
            }
        }
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let value: u32 = unsafe { transmute(buf) };
        if value == u32::MAX {
            Ok(NodePersistRef::Invalid)
        } else {
            Ok(NodePersistRef::Reference(Box::new(
                NodePersist::deserialize(reader, value)?,
            )))
        }
    }
}

impl CustomSerialize for PropPersistRef {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        writer.write_u32::<LittleEndian>(self.0)?; // Write offset
        writer.write_u32::<LittleEndian>(self.1)?; // Write length
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let prop_offset = reader.read_u32::<LittleEndian>()?;
        let prop_length = reader.read_u32::<LittleEndian>()?;
        Ok((prop_offset, prop_length))
    }
}
impl CustomSerialize for NeighbourPersist {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        self.node.serialize(writer)?;
        writer.write(unsafe { transmute::<f32, [u8; 4]>(self.cosine_similarity).as_ref() })?;
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let node_offset = reader.stream_position()? as u32;
        let node = NodePersistRef::deserialize(reader, node_offset)?;
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let cosine_similarity: f32 = unsafe { transmute(buf) };
        Ok(NeighbourPersist {
            node,
            cosine_similarity,
        })
    }
}

impl CustomSerialize for VersionRef {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        match self {
            VersionRef::Reference(versions) => {
                writer.write_u32::<LittleEndian>(0)?;
                let versions_offset = versions.serialize(writer)?;
                println!(
                    "Versions | offset: {}, versions_offset: {}",
                    offset, versions_offset
                );
                writer.seek(SeekFrom::Start(offset as u64))?;
                writer.write_u32::<LittleEndian>(versions_offset)?;
                writer.seek(SeekFrom::End(0))?;
            }
            VersionRef::Invalid => writer.write_u32::<LittleEndian>(std::u32::MAX)?,
        };
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let value = reader.read_u32::<LittleEndian>()?;
        println!("Version ref value: {}", value);
        if value == std::u32::MAX {
            Ok(VersionRef::Invalid)
        } else {
            Ok(VersionRef::Reference(Box::new(Versions::deserialize(
                reader, value,
            )?)))
        }
    }
}
impl CustomSerialize for Versions {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        println!("Versions @@ offset: {}", offset);
        for version in &self.versions {
            println!("Versions serialize");
            version.serialize(writer)?;
        }
        self.next.serialize(writer)?;
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        // Get all offsets first
        let offsets = [
            reader.stream_position()? as u32,
            reader.stream_position()? as u32,
            reader.stream_position()? as u32,
            reader.stream_position()? as u32,
        ];

        println!("Versions offsets: {:?}", offsets);
        // Now deserialize using the offsets
        let versions = [
            NodePersistRef::deserialize(reader, offsets[0])?,
            NodePersistRef::deserialize(reader, offsets[0])?,
            NodePersistRef::deserialize(reader, offsets[0])?,
            NodePersistRef::deserialize(reader, offsets[1])?,
        ];

        let next_offset = reader.stream_position()? as u32;
        let next = VersionRef::deserialize(reader, next_offset)?;

        Ok(Versions { versions, next })
    }
}
impl CustomSerialize for NodePersist {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let st_offset = writer.stream_position()? as u32;

        // Serialize version_id
        writer.write_u16::<LittleEndian>(self.version_id)?;

        // Serialize prop_location
        self.prop_location.serialize(writer)?;

        // Serialize hnsw_level
        writer.write_u8(self.hnsw_level)?;

        // Combine parent and child indicators into a single byte
        let mut indicator: u8 = 0;
        if self.parent.is_some() {
            indicator |= 0b01;
        }
        if self.child.is_some() {
            indicator |= 0b10;
        }
        writer.write_u8(indicator)?;

        // Serialize parent if present
        if let Some(parent) = &self.parent {
            parent.serialize(writer)?;
        }

        // Serialize child if present
        if let Some(child) = &self.child {
            child.serialize(writer)?;
        }

        // Serialize version_ref
        self.version_ref.serialize(writer)?;

        // Serialize neighbors
        for neighbor in &self.neighbors {
            neighbor.serialize(writer)?;
        }
        let end_offset = writer.stream_position()? as u32;
        println!(
            "full length: {}, parent {:?}, child {:?}",
            end_offset - st_offset,
            self.parent,
            self.child
        );

        Ok(st_offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        println!("Starting deserialization at offset: {}", offset);
        reader.seek(SeekFrom::Start(offset as u64))?;

        let version_id = reader.read_u16::<LittleEndian>()?;
        println!("Read version_id: {}", version_id);

        let prop_location_offset = reader.stream_position()? as u32;
        println!("Prop location offset: {}", prop_location_offset);
        let prop_location = PropPersistRef::deserialize(reader, prop_location_offset)?;
        println!("Deserialized prop_location");

        let hnsw_level = reader.read_u8()?;
        println!("Read hnsw_level: {}", hnsw_level);
        // Read the combined indicator byte
        let indicator = reader.read_u8()?;
        println!("Read indicator byte: {:08b}", indicator);

        let parent = if indicator & 0b01 != 0 {
            let parent_offset = reader.stream_position()? as u32;
            println!("Deserializing parent at offset: {}", parent_offset);
            Some(NodePersistRef::deserialize(reader, parent_offset)?)
        } else {
            println!("No parent present");
            None
        };

        let child = if indicator & 0b10 != 0 {
            let child_offset = reader.stream_position()? as u32;
            println!("Deserializing child at offset: {}", child_offset);
            Some(NodePersistRef::deserialize(reader, child_offset)?)
        } else {
            println!("No child present");
            None
        };

        let version_ref_offset = reader.stream_position()? as u32;
        println!("Version ref offset: {}", version_ref_offset);
        let version_ref = VersionRef::deserialize(reader, version_ref_offset)?;
        println!("Deserialized version_ref");

        // Deserialize neighbors
        println!("Starting to deserialize neighbors");
        let mut neighbors = Vec::with_capacity(10);
        for i in 0..10 {
            let neighbor_offset = reader.stream_position()? as u32;
            println!(
                "Deserializing neighbor {} at offset: {}",
                i, neighbor_offset
            );
            neighbors.push(NeighbourPersist::deserialize(reader, neighbor_offset)?);
        }
        let neighbors: [NeighbourPersist; 10] = neighbors.try_into().unwrap();
        println!("Finished deserializing neighbors");

        println!("Deserialization completed successfully");
        Ok(NodePersist {
            version_id,
            prop_location,
            hnsw_level,
            version_ref,
            neighbors,
            parent,
            child,
        })
    }
}
