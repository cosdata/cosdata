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
        let parent_present = !matches!(*self.parent.read().unwrap(), LazyItem::Null);
        let child_present = !matches!(*self.child.read().unwrap(), LazyItem::Null);
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
            let offset = writer.stream_position()? as u32;
            self.parent.read().unwrap().serialize(writer)?;
            Some(offset)
        } else {
            None
        };

        // Serialize child if present
        let child_offset = if child_present {
            let offset = writer.stream_position()? as u32;
            self.child.read().unwrap().serialize(writer)?;
            Some(offset)
        } else {
            None
        };

        // Serialize neighbors
        let neighbors_offset = writer.stream_position()? as u32;
        self.neighbors.read().unwrap().serialize(writer)?;

        // Serialize versions
        let versions_offset = writer.stream_position()? as u32;
        self.versions.read().unwrap().serialize(writer)?;

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

        // Create LazyItems for parent and child
        let parent = if let Some(offset) = parent_offset {
            LazyItem::LazyLoad(offset)
        } else {
            LazyItem::Null
        };

        let child = if let Some(offset) = child_offset {
            LazyItem::LazyLoad(offset)
        } else {
            LazyItem::Null
        };

        // Create LazyItems for neighbors and versions
        let neighbors = LazyItems::deserialize(reader, neighbors_offset)?;
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
        let offset = match self {
            LazyItem::Ready(item, Some(existing_offset)) => {
                writer.seek(SeekFrom::Start(*existing_offset as u64))?;
                item.serialize(writer)?;
                *existing_offset
            }
            LazyItem::Ready(item, None) => {
                let offs = item.serialize(writer)?;
                offs
            }
            LazyItem::LazyLoad(file_offset) => *file_offset,
            LazyItem::Null => u32::MAX,
        };

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let item = T::deserialize(reader, offset)?;
        Ok(LazyItem::Ready(Arc::new(item), Some(offset)))
    }
}

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItems<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let start_offset = writer.stream_position()? as u32;

        // Write the number of items
        writer.write_u32::<LittleEndian>(self.items.len() as u32)?;

        let mut current_chunk_start = writer.stream_position()? as u32;
        for chunk in self.items.chunks(CHUNK_SIZE) {
            // Write placeholders for item offsets
            let placeholder_start = writer.stream_position()? as u32;
            for _ in 0..CHUNK_SIZE {
                writer.write_u32::<LittleEndian>(u32::MAX)?;
            }

            // Write placeholder for next chunk link
            let next_chunk_placeholder = writer.stream_position()? as u32;
            writer.write_u32::<LittleEndian>(u32::MAX)?;

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
            writer.seek(SeekFrom::Start(next_chunk_placeholder as u64))?;
            if next_chunk_start == current_chunk_start {
                writer.write_u32::<LittleEndian>(u32::MAX)?; // Last chunk
            } else {
                writer.write_u32::<LittleEndian>(next_chunk_start)?;
            }
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
                    items.push(LazyItem::LazyLoad(item_offset));
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
