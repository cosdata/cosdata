use crate::models::file_persist::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

pub trait CustomSerialize {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized;
}

impl CustomSerialize for NodePersistRef {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.0)?;
        writer.write_u32::<LittleEndian>(self.1)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let file_number = reader.read_u32::<LittleEndian>()?;
        let offset = reader.read_u32::<LittleEndian>()?;
        Ok((file_number, offset))
    }
}

impl CustomSerialize for NeighbourPersist {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.node.serialize(writer)?;
        writer.write_f32::<LittleEndian>(self.cosine_similarity)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let node = NodePersistRef::deserialize(reader)?;
        let cosine_similarity = reader.read_f32::<LittleEndian>()?;
        Ok(NeighbourPersist {
            node,
            cosine_similarity,
        })
    }
}

impl CustomSerialize for VersionRef {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for version in &self.versions {
            version.serialize(writer)?;
        }
        self.next.serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut versions = [NodePersistRef::default(); 4];
        for version in &mut versions {
            *version = NodePersistRef::deserialize(reader)?;
        }
        let next = NodePersistRef::deserialize(reader)?;
        Ok(VersionRef { versions, next })
    }
}

impl CustomSerialize for NodePersist {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.version_id)?;
        match &self.version_ref {
            Some(ver_ref) => {
                writer.write_u8(1)?;
                ver_ref.serialize(writer)?;
            }
            None => writer.write_u8(0)?,
        }
        self.prop_location.serialize(writer)?;
        writer.write_u8(self.hnsw_level)?;
        for neighbor in &self.neighbors {
            neighbor.serialize(writer)?;
        }
        match &self.parent {
            Some(parent) => {
                writer.write_u8(1)?;
                parent.serialize(writer)?;
            }
            None => writer.write_u8(0)?,
        }
        match &self.child {
            Some(child) => {
                writer.write_u8(1)?;
                child.serialize(writer)?;
            }
            None => writer.write_u8(0)?,
        }
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let version_id = reader.read_u32::<LittleEndian>()?;
        let version_ref = if reader.read_u8()? == 1 {
            Some(VersionRef::deserialize(reader)?)
        } else {
            None
        };
        let prop_location = NodePersistRef::deserialize(reader)?;
        let hnsw_level = reader.read_u8()?;

        let mut neighbors = Vec::with_capacity(20);
        for _ in 0..20 {
            neighbors.push(NeighbourPersist::deserialize(reader)?);
        }
        let neighbors: [NeighbourPersist; 20] = neighbors.try_into().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to create neighbors array",
            )
        })?;

        let parent = if reader.read_u8()? == 1 {
            Some(NodePersistRef::deserialize(reader)?)
        } else {
            None
        };
        let child = if reader.read_u8()? == 1 {
            Some(NodePersistRef::deserialize(reader)?)
        } else {
            None
        };
        Ok(NodePersist {
            version_id,
            version_ref,
            prop_location,
            hnsw_level,
            neighbors,
            parent,
            child,
        })
    }
}

// use std::io::Cursor;

// fn test() -> std::io::Result<()> {
//     let node = NodePersist {
//         // ... initialize fields ...
//     };

//     let mut buffer = Vec::new();
//     node.serialize(&mut buffer)?;

//     // Deserialize
//     let mut cursor = Cursor::new(buffer);
//     let deserialized_node = NodePersist::deserialize(&mut cursor)?;

//     Ok(())
// }
