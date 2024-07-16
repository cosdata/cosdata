use crate::models::file_persist::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};
use std::mem::{size_of, transmute};

pub trait CustomSerialize {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized;
}

impl CustomSerialize for NodePersistRef {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(unsafe { transmute::<_, &[u8; 8]>(self) })
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(unsafe { transmute(buf) })
    }
}

impl CustomSerialize for NeighbourPersist {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(unsafe { transmute::<_, &[u8; 12]>(self) })
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 12];
        reader.read_exact(&mut buf)?;
        Ok(unsafe { transmute(buf) })
    }
}

impl CustomSerialize for VersionRef {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(unsafe { transmute::<_, &[u8; 40]>(self) })
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 40];
        reader.read_exact(&mut buf)?;
        Ok(unsafe { transmute(buf) })
    }
}

impl CustomSerialize for NodePersist {
    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.version_id)?;

        let flags = (self.version_ref.is_some() as u8)
            | ((self.parent.is_some() as u8) << 1)
            | ((self.child.is_some() as u8) << 2);
        writer.write_u8(flags)?;

        if let Some(ver_ref) = &self.version_ref {
            ver_ref.serialize(writer)?;
        }

        self.prop_location.serialize(writer)?;
        writer.write_u8(self.hnsw_level)?;

        // Serialize neighbors as a single block
        writer.write_all(unsafe {
            transmute::<_, &[u8; 10 * size_of::<NeighbourPersist>()]>(&self.neighbors)
        })?;

        if let Some(parent) = &self.parent {
            parent.serialize(writer)?;
        }
        if let Some(child) = &self.child {
            child.serialize(writer)?;
        }
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let version_id = reader.read_u32::<LittleEndian>()?;
        let flags = reader.read_u8()?;

        let version_ref = if flags & 1 != 0 {
            Some(VersionRef::deserialize(reader)?)
        } else {
            None
        };

        let prop_location = NodePersistRef::deserialize(reader)?;
        let hnsw_level = reader.read_u8()?;

        let mut neighbors = [NeighbourPersist {
            node: (0, 0),
            cosine_similarity: 0.0,
        }; 10];
        reader.read_exact(unsafe {
            transmute::<_, &mut [u8; 10 * size_of::<NeighbourPersist>()]>(&mut neighbors)
        })?;

        let parent = if flags & 2 != 0 {
            Some(NodePersistRef::deserialize(reader)?)
        } else {
            None
        };

        let child = if flags & 4 != 0 {
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
