use super::CustomSerialize;
use crate::models::{
    cache_loader::NodeRegistry,
    chunked_list::{LazyItem, LazyItems, CHUNK_SIZE},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, RwLock},
};

use crate::models::types::FileOffset;
use std::collections::HashSet;
impl<T> CustomSerialize for LazyItems<T>
where
    LazyItem<T>: CustomSerialize,
    T: Clone,
{
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        if self.is_empty() {
            return Ok(u32::MAX);
        };
        let start_offset = writer.stream_position()? as u32;
        let mut items_guard = self.items.write().unwrap();
        let total_items = items_guard.len();

        for chunk_start in (0..total_items).step_by(CHUNK_SIZE) {
            let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, total_items);
            let is_last_chunk = chunk_end == total_items;

            // Write placeholders for item offsets
            let placeholder_start = writer.stream_position()? as u32;
            for _ in 0..CHUNK_SIZE {
                writer.write_u32::<LittleEndian>(u32::MAX)?;
            }
            // Write placeholder for next chunk link
            let next_chunk_placeholder = writer.stream_position()? as u32;
            writer.write_u32::<LittleEndian>(u32::MAX)?;

            // Serialize items and update placeholders
            for i in chunk_start..chunk_end {
                let item_offset = items_guard[i].serialize(writer)?;
                items_guard[i].offset = Some(FileOffset(item_offset));
                let placeholder_pos = placeholder_start as u64 + ((i - chunk_start) as u64 * 4);
                let current_pos = writer.stream_position()?;
                writer.seek(SeekFrom::Start(placeholder_pos))?;
                writer.write_u32::<LittleEndian>(item_offset)?;
                writer.seek(SeekFrom::Start(current_pos))?;
            }

            // Write next chunk link
            let next_chunk_start = writer.stream_position()? as u32;
            writer.seek(SeekFrom::Start(next_chunk_placeholder as u64))?;
            if is_last_chunk {
                writer.write_u32::<LittleEndian>(u32::MAX)?; // Last chunk
            } else {
                writer.write_u32::<LittleEndian>(next_chunk_start)?;
            }
            writer.seek(SeekFrom::Start(next_chunk_start as u64))?;
        }
        Ok(start_offset)
    }
    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: FileOffset,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self> {
        if offset.0 == u32::MAX {
            return Ok(LazyItems::new());
        }
        reader.seek(SeekFrom::Start(offset.0 as u64))?;
        let mut items = Vec::new();
        let FileOffset(mut current_chunk) = offset;
        loop {
            for i in 0..CHUNK_SIZE {
                reader.seek(SeekFrom::Start(current_chunk as u64 + (i as u64 * 4)))?;
                let item_offset = reader.read_u32::<LittleEndian>()?;
                if item_offset == u32::MAX {
                    continue;
                }
                let item = LazyItem::deserialize(
                    reader,
                    FileOffset(item_offset),
                    cache.clone(),
                    max_loads,
                    skipm,
                )?;
                items.push(item);
            }
            reader.seek(SeekFrom::Start(
                current_chunk as u64 + CHUNK_SIZE as u64 * 4,
            ))?;
            // Read next chunk link
            current_chunk = reader.read_u32::<LittleEndian>()?;
            if current_chunk == u32::MAX {
                break;
            }
        }
        Ok(LazyItems {
            items: Arc::new(RwLock::new(items)),
        })
    }
}
