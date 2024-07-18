use crate::models::file_persist::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::{size_of, transmute};

pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized;
}

impl CustomSerialize for NodePersistRef {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        writer.write_all(unsafe { transmute::<_, &[u8; 4]>(self) })?;
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(unsafe { transmute(buf) })
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
        writer.write_all(unsafe { transmute::<_, &[u8; 8]>(self) })?;
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(unsafe { transmute(buf) })
    }
}

impl CustomSerialize for VersionRef {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        match self {
            VersionRef::Reference(versions) => {
                writer.write_u32::<LittleEndian>(0)?;
                let versions_offset = versions.serialize(writer)?;
                writer.seek(SeekFrom::Start(offset as u64))?;
                writer.write_u32::<LittleEndian>(versions_offset)?;
                writer.seek(SeekFrom::End(0))?;
            }
            VersionRef::Invalid(_) => writer.write_u32::<LittleEndian>(std::u32::MAX)?,
        };
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let value = reader.read_u32::<LittleEndian>()?;
        if value == std::u32::MAX {
            Ok(VersionRef::Invalid(0))
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
        writer.write_all(unsafe { transmute::<_, &[u8; 16]>(&self.versions) })?;
        let next_offset_pos = writer.stream_position()? as u32;
        writer.write_u32::<LittleEndian>(0)?;

        match &self.next {
            VersionRef::Reference(versions) => {
                let next_offset = versions.serialize(writer)?;
                writer.seek(SeekFrom::Start(next_offset_pos as u64))?;
                writer.write_u32::<LittleEndian>(next_offset)?;
                writer.seek(SeekFrom::End(0))?;
            }
            VersionRef::Invalid(_) => {
                writer.seek(SeekFrom::Start(next_offset_pos as u64))?;
                writer.write_u32::<LittleEndian>(std::u32::MAX)?;
                writer.seek(SeekFrom::End(0))?;
            }
        }
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let mut versions = [0; 4];
        reader.read_exact(unsafe { transmute::<_, &mut [u8; 16]>(&mut versions) })?;
        let next_offset = reader.read_u32::<LittleEndian>()?;
        let next = if next_offset == std::u32::MAX {
            VersionRef::Invalid(0)
        } else {
            VersionRef::Reference(Box::new(Versions::deserialize(reader, next_offset)?))
        };
        Ok(Versions { versions, next })
    }
}

impl CustomSerialize for NodePersist {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;
        writer.write_u32::<LittleEndian>(self.version_id)?;
        self.prop_location.serialize(writer)?;
        writer.write_u8(self.hnsw_level)?;
        self.version_ref.serialize(writer)?;

        writer.write_all(unsafe { transmute::<_, &[u8; 80]>(&self.neighbors) })?;

        let mut flags = 0u8;
        if self.parent.is_some() {
            flags |= 1;
        }
        if self.child.is_some() {
            flags |= 2;
        }
        writer.write_u8(flags)?;

        if let Some(parent) = self.parent {
            parent.serialize(writer)?;
        }
        if let Some(child) = self.child {
            child.serialize(writer)?;
        }

        Ok(offset)
    }
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let version_id = reader.read_u32::<LittleEndian>()?;

        let prop_location_offset = reader.stream_position()? as u32;
        let prop_location = PropPersistRef::deserialize(reader, prop_location_offset)?;

        let hnsw_level = reader.read_u8()?;

        let version_ref_offset = reader.stream_position()? as u32;
        let version_ref = VersionRef::deserialize(reader, version_ref_offset)?;

        let mut neighbors = [NeighbourPersist {
            node: 0,
            cosine_similarity: 0.0,
        }; 10];
        reader.read_exact(unsafe { transmute::<_, &mut [u8; 80]>(&mut neighbors) })?;

        let flags = reader.read_u8()?;

        let parent = if flags & 1 != 0 {
            let parent_offset = reader.stream_position()? as u32;
            Some(NodePersistRef::deserialize(reader, parent_offset)?)
        } else {
            None
        };

        let child = if flags & 2 != 0 {
            let child_offset = reader.stream_position()? as u32;
            Some(NodePersistRef::deserialize(reader, child_offset)?)
        } else {
            None
        };

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
