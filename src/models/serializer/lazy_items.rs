use super::CustomSerialize;
use crate::models::chunked_list::{LazyItem, LazyItems, CHUNK_SIZE};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItems<T> {
    fn serialize<W: Write + Seek>(&mut self, writer: &mut W) -> std::io::Result<u32> {
        let start_offset = writer.stream_position()? as u32;

        // Write the number of items
        writer.write_u32::<LittleEndian>(self.items.len() as u32)?;

        let mut current_chunk_start = writer.stream_position()? as u32;
        for chunk in self.items.chunks_mut(CHUNK_SIZE) {
            // Write placeholders for item offsets
            let placeholder_start = writer.stream_position()? as u32;
            for _ in 0..CHUNK_SIZE {
                writer.write_u32::<LittleEndian>(u32::MAX)?;
            }

            // Write placeholder for next chunk link
            let next_chunk_placeholder = writer.stream_position()? as u32;
            writer.write_u32::<LittleEndian>(u32::MAX)?;

            // Serialize items and update placeholders
            for (i, item) in chunk.iter_mut().enumerate() {
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
