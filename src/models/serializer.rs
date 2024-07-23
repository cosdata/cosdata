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
        self.serialize(writer)?;

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
        let start_offset = writer.stream_position()? as u32;

        // Serialize basic fields
        writer.write_u16::<LittleEndian>(self.version_id)?;
        writer.write_u8(self.hnsw_level)?;

        // // Serialize prop
        //self.prop.read().unwrap().serialize(writer)?;

        // Serialize neighbors
        self.neighbors.read().unwrap().serialize(writer)?;

        // Serialize parent and child
        self.parent.read().unwrap().serialize(writer)?;
        self.child.read().unwrap().serialize(writer)?;

        // Serialize versions
        self.versions.read().unwrap().serialize(writer)?;

        Ok(start_offset)
    }
    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let version_id = reader.read_u16::<LittleEndian>()?;
        let hnsw_level = reader.read_u8()?;

        // Read prop offset
        let prop_offset = reader.read_u32::<LittleEndian>()?;
        let prop_length = reader.read_u32::<LittleEndian>()?;

        // Read other offsets
        let neighbors_offset = reader.read_u32::<LittleEndian>()?;
        let parent_offset = reader.read_u32::<LittleEndian>()?;
        let child_offset = reader.read_u32::<LittleEndian>()?;
        let versions_offset = reader.read_u32::<LittleEndian>()?;

        // Now deserialize each component
        let prop = PropState::Pending((prop_offset, prop_length));
        let neighbors = LazyItems::deserialize(reader, neighbors_offset)?;
        let parent = LazyItem::deserialize(reader, parent_offset)?;
        let child = LazyItem::deserialize(reader, child_offset)?;
        let versions = LazyItems::deserialize(reader, versions_offset)?;

        Ok(MergedNode {
            version_id,
            hnsw_level,
            prop: Arc::new(RwLock::new(prop)),
            neighbors: Arc::new(RwLock::new(neighbors)),
            parent: Arc::new(RwLock::new(parent)),
            child: Arc::new(RwLock::new(child)),
            versions: Arc::new(RwLock::new(versions)),
            persist_flag: Arc::new(RwLock::new(true)),
        })
    }
}

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItem<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        match self {
            LazyItem::Ready(item, Some(existing_offset)) => {
                writer.write_u32::<LittleEndian>(*existing_offset)?;
                writer.seek(SeekFrom::Start(*existing_offset as u64))?;
                item.serialize(writer)?;
            }
            LazyItem::Ready(item, None) => {
                let item_offset = offset + 4;
                writer.write_u32::<LittleEndian>(item_offset)?;
                item.serialize(writer)?;
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

        let item_offset = reader.read_u32::<LittleEndian>()?;

        if item_offset == u32::MAX {
            Ok(LazyItem::Null)
        } else if item_offset == offset + 4 {
            let item = T::deserialize(reader, item_offset)?;
            Ok(LazyItem::Ready(Arc::new(item), Some(item_offset)))
        } else {
            Ok(LazyItem::LazyLoad(item_offset))
        }
    }
}

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItems<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let start_offset = writer.stream_position()? as u32;

        // Write the number of items
        writer.write_u32::<LittleEndian>(self.items.len() as u32)?;

        let mut current_chunk_start = writer.stream_position()? as u32;

        // Serialize items
        for (chunk_index, chunk) in self.items.chunks(CHUNK_SIZE).enumerate() {
            // Write placeholders + next chunk link
            let placeholder_start = writer.stream_position()? as u32;
            for _ in 0..CHUNK_SIZE + 1 {
                writer.write_u32::<LittleEndian>(u32::MAX)?;
            }

            // Serialize items and update placeholders
            for (i, item) in chunk.iter().enumerate() {
                let item_offset = item.serialize(writer)?;
                let placeholder_pos = placeholder_start as u64 + (i as u64 * 4);
                let current_pos = writer.stream_position()?;
                writer.seek(SeekFrom::Start(placeholder_pos))?;
                writer.write_u32::<LittleEndian>(item_offset)?;
                writer.seek(SeekFrom::Start(current_pos))?;
            }

            // Write next chunk link
            let next_chunk_start = writer.stream_position()? as u32;
            writer.seek(SeekFrom::Start(
                (current_chunk_start + (CHUNK_SIZE as u32 * 4)) as u64,
            ))?;
            writer.write_u32::<LittleEndian>(next_chunk_start)?;
            writer.seek(SeekFrom::Start(next_chunk_start as u64))?;

            current_chunk_start = next_chunk_start;
        }

        Ok(start_offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let item_count = reader.read_u32::<LittleEndian>()? as usize;
        let mut items = Vec::with_capacity(item_count);

        let mut current_chunk = reader.stream_position()? as u32;
        while items.len() < item_count {
            reader.seek(SeekFrom::Start(current_chunk as u64))?;
            for _ in 0..CHUNK_SIZE {
                if items.len() >= item_count {
                    break;
                }
                let item_offset = reader.read_u32::<LittleEndian>()?;
                if item_offset == u32::MAX {
                    items.push(LazyItem::Null);
                } else {
                    items.push(LazyItem::deserialize(reader, item_offset)?);
                }
            }
            // Read next chunk link
            current_chunk = reader.read_u32::<LittleEndian>()?;
            if current_chunk == u32::MAX {
                break;
            }
        }

        Ok(LazyItems { items })
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
                    for value in inner_vec {
                        writer.write_u32::<LittleEndian>(*value)?;
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
