use crate::models::chunked_list::*;
use crate::models::file_persist::*;
use crate::models::types::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::transmute;

use std::sync::{Arc, Mutex, OnceLock, RwLock};
pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized;
}

impl CustomSerialize for Neighbour {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        // Serialize the node
        self.node.serialize(writer)?;

        // Serialize the cosine similarity
        writer.write_f32::<LittleEndian>(self.cosine_similarity)?;

        Ok(offset)
    }
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        // Deserialize the node
        let node = MergedNode::deserialize(reader, offset)?;

        // Deserialize the cosine similarity
        let cosine_similarity = reader.read_f32::<LittleEndian>()?;

        Ok(Neighbour {
            node: Arc::new(node),
            cosine_similarity,
        })
    }
}

impl CustomSerialize for MergedNode {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        writer.write_u16::<LittleEndian>(self.version_id)?;
        writer.write_u8(self.hnsw_level)?;

        // Serialize PropState
        match &*self.prop.read().unwrap() {
            PropState::Ready(node_prop) => {
                if let Some(location) = node_prop.location {
                    writer.write_u32::<LittleEndian>(location.0)?;
                    writer.write_u32::<LittleEndian>(location.1)?;
                } else {
                    // Handle the case where location is None
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Ready PropState has no location",
                    ));
                }
            }
            PropState::Pending(prop_ref) => {
                writer.write_u32::<LittleEndian>(prop_ref.0)?;
                writer.write_u32::<LittleEndian>(prop_ref.1)?;
            }
        }

        // Skip Serializing location , redundant cyclic

        // Write placeholders for parent and child offsets
        let parent_placeholder_pos = writer.stream_position()? as u32;
        match &*self.parent.read().unwrap() {
            LazyItem::Null => writer.write_u32::<LittleEndian>(u32::MAX)?,
            _ => writer.write_u32::<LittleEndian>(0)?, // Temporary placeholder
        }

        let child_placeholder_pos = writer.stream_position()? as u32;
        match &*self.child.read().unwrap() {
            LazyItem::Null => writer.write_u32::<LittleEndian>(u32::MAX)?,
            _ => writer.write_u32::<LittleEndian>(0)?, // Temporary placeholder
        }

        // Queue parent and child for serialization
        let mut serialization_queue = Vec::new();
        if let LazyItem::Ready(parent) = &*self.parent.read().unwrap() {
            serialization_queue.push((parent_placeholder_pos, Arc::clone(parent)));
        }
        if let LazyItem::Ready(child) = &*self.child.read().unwrap() {
            serialization_queue.push((child_placeholder_pos, Arc::clone(child)));
        }

        // Serialize neighbors
        self.neighbors.read().unwrap().serialize(writer)?;

        // Serialize version_ref
        self.version_ref.read().unwrap().serialize(writer)?;

        // Serialize queued nodes and update placeholders
        for (placeholder_pos, node) in serialization_queue {
            let node_offset = node.serialize(writer)?;

            // Update placeholder
            let end_pos = writer.stream_position()?;
            writer.seek(SeekFrom::Start(placeholder_pos as u64))?;
            writer.write_u32::<LittleEndian>(node_offset)?;
            writer.seek(SeekFrom::Start(end_pos))?;
        }

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let version_id = reader.read_u16::<LittleEndian>()?;
        let hnsw_level = reader.read_u8()?;

        // Deserialize PropState
        let prop = {
            let offset = reader.read_u32::<LittleEndian>()?;
            let length = reader.read_u32::<LittleEndian>()?;

            // We don't know if it's Ready or Pending at this point, so we'll assume Pending
            // The actual state will be determined when the prop is accessed
            PropState::Pending((offset, length))
        };

        // we just take the offset
        let location = offset;
        // Read offsets for neighbors, parent, child, and version_ref
        let neighbors_offset = reader.read_u32::<LittleEndian>()?;
        let parent_offset = reader.read_u32::<LittleEndian>()?;
        let child_offset = reader.read_u32::<LittleEndian>()?;
        let version_ref_offset = reader.read_u32::<LittleEndian>()?;

        let parent = if parent_offset == u32::MAX {
            LazyItem::Null
        } else {
            LazyItem::LazyLoad(parent_offset)
        };
        let child = if child_offset == u32::MAX {
            LazyItem::Null
        } else {
            LazyItem::LazyLoad(child_offset)
        };

        // Deserialize neighbors
        let neighbors = ItemListRef::deserialize(reader, neighbors_offset)?;

        // Deserialize version_ref
        let version_ref = ItemListRef::deserialize(reader, version_ref_offset)?;

        Ok(MergedNode {
            version_id,
            hnsw_level,
            prop: Arc::new(RwLock::new(prop)),
            location: Arc::new(RwLock::new(Some(location))),
            neighbors: Arc::new(RwLock::new(neighbors)),
            parent: Arc::new(RwLock::new(parent)),
            child: Arc::new(RwLock::new(child)),
            version_ref: Arc::new(RwLock::new(version_ref)),
            persist_flag: Arc::new(RwLock::new(true)),
        })
    }
}

impl<T: Clone + Locatable + CustomSerialize> CustomSerialize for LazyItem<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        match self {
            LazyItem::Ready(item) => {
                let location = item.get_file_offset().unwrap_or(u32::MAX);

                writer.write_u32::<LittleEndian>(location)?;

                if item.needs_persistence() {
                    item.serialize(writer)?;
                }
            }
            LazyItem::LazyLoad(file_offset) => {
                writer.write_u32::<LittleEndian>(*file_offset)?;
            }
            LazyItem::Null => {
                writer.write_u32::<LittleEndian>(u32::MAX)?;
            }
        }

        Ok(offset)
    }
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let location = reader.read_u32::<LittleEndian>()?;

        if location == u32::MAX {
            Ok(LazyItem::Null)
        } else {
            // We're not reading the actual T here, just its location
            Ok(LazyItem::LazyLoad(location))
        }
    }
}

impl<T: Clone + Locatable + CustomSerialize> CustomSerialize for Items<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        for item in &self.items {
            item.serialize(writer)?;
        }

        self.next.serialize(writer)?;

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        // Read all offsets first
        let offset1 = reader.stream_position()? as u32;
        reader.seek(SeekFrom::Current(4))?; // Assuming each LazyItem is 4 bytes
        let offset2 = reader.stream_position()? as u32;
        reader.seek(SeekFrom::Current(4))?;
        let offset3 = reader.stream_position()? as u32;
        reader.seek(SeekFrom::Current(4))?;
        let offset4 = reader.stream_position()? as u32;
        reader.seek(SeekFrom::Current(4))?;
        let next_offset = reader.stream_position()? as u32;

        // Now deserialize using the offsets
        let items = [
            LazyItem::deserialize(reader, offset1)?,
            LazyItem::deserialize(reader, offset2)?,
            LazyItem::deserialize(reader, offset3)?,
            LazyItem::deserialize(reader, offset4)?,
        ];

        let next = ItemListRef::deserialize(reader, next_offset)?;

        Ok(Items { items, next })
    }
}

impl<T: Clone + Locatable + CustomSerialize> CustomSerialize for ItemListRef<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        match self {
            ItemListRef::Ref(items) => {
                writer.write_u32::<LittleEndian>(offset + 4)?; // Point to the next 4 bytes
                items.serialize(writer)?;
            }
            ItemListRef::Null => {
                writer.write_u32::<LittleEndian>(u32::MAX)?;
            }
        }

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let next_offset = reader.read_u32::<LittleEndian>()?;

        if next_offset == u32::MAX {
            Ok(ItemListRef::Null)
        } else {
            // Deserialize Items directly here
            let items = Items::deserialize(reader, next_offset)?;
            Ok(ItemListRef::Ref(Box::new(items)))
        }
    }
}

impl CustomSerialize for VectorQt {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        match self {
            VectorQt::UnsignedByte { mag, quant_vec } => {
                writer.write_u8(0)?; // 0 for UnsignedByte
                writer.write_u32::<LittleEndian>(*mag)?;
                writer.write_u32::<LittleEndian>(quant_vec.len() as u32)?;
                writer.write_all(quant_vec)?;
            }
            VectorQt::SubByte {
                mag,
                quant_vec,
                resolution,
            } => {
                writer.write_u8(1)?; // 1 for SubByte
                writer.write_u32::<LittleEndian>(*mag)?;
                writer.write_u32::<LittleEndian>(quant_vec.len() as u32)?;
                for inner_vec in quant_vec {
                    writer.write_u32::<LittleEndian>(inner_vec.len() as u32)?;
                    for &value in inner_vec {
                        writer.write_u32::<LittleEndian>(value)?;
                    }
                }
                writer.write_u8(*resolution)?;
            }
        }

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        match reader.read_u8()? {
            0 => {
                let mag = reader.read_u32::<LittleEndian>()?;
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let mut quant_vec = vec![0u8; len];
                reader.read_exact(&mut quant_vec)?;
                Ok(VectorQt::UnsignedByte { mag, quant_vec })
            }
            1 => {
                let mag = reader.read_u32::<LittleEndian>()?;
                let outer_len = reader.read_u32::<LittleEndian>()? as usize;
                let mut quant_vec = Vec::with_capacity(outer_len);
                for _ in 0..outer_len {
                    let inner_len = reader.read_u32::<LittleEndian>()? as usize;
                    let mut inner_vec = Vec::with_capacity(inner_len);
                    for _ in 0..inner_len {
                        inner_vec.push(reader.read_u32::<LittleEndian>()?);
                    }
                    quant_vec.push(inner_vec);
                }
                let resolution = reader.read_u8()?;
                Ok(VectorQt::SubByte {
                    mag,
                    quant_vec,
                    resolution,
                })
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Null VectorQt",
            )),
        }
    }
}
