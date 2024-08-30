use super::CustomSerialize;
use crate::models::lazy_load::{
    EagerLazyItem, EagerLazyItemSet, FileIndex, LazyItem, SyncPersist, CHUNK_SIZE,
};
use crate::models::types::FileOffset;
use crate::models::{
    cache_loader::NodeRegistry,
    identity_collections::{Identifiable, IdentitySet},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

impl<T, E> CustomSerialize for EagerLazyItemSet<T, E>
where
    LazyItem<T>: CustomSerialize,
    T: Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        if self.is_empty() {
            return Ok(u32::MAX);
        };
        let start_offset = writer.stream_position()? as u32;
        let mut items_arc = self.items.clone();
        let items: Vec<_> = items_arc.get().iter().map(Clone::clone).collect();
        let total_items = items.len();

        for chunk_start in (0..total_items).step_by(CHUNK_SIZE) {
            let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, total_items);
            let is_last_chunk = chunk_end == total_items;

            // Write placeholders for item offsets
            let placeholder_start = writer.stream_position()? as u32;
            for _ in 0..CHUNK_SIZE {
                writer.write_u32::<LittleEndian>(u32::MAX)?;
                writer.write_u32::<LittleEndian>(u32::MAX)?;
            }
            // Write placeholder for next chunk link
            let next_chunk_placeholder = writer.stream_position()? as u32;
            writer.write_u32::<LittleEndian>(u32::MAX)?;

            // Serialize items and update placeholders
            for i in chunk_start..chunk_end {
                let item_offset = items[i].serialize(writer)?;
                let placeholder_pos = placeholder_start as u64 + ((i - chunk_start) as u64 * 8);
                let current_pos = writer.stream_position()?;
                writer.seek(SeekFrom::Start(placeholder_pos))?;
                writer.write_u32::<LittleEndian>(item_offset)?;
                writer.write_u32::<LittleEndian>(*items[i].1.get_current_version())?;
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
        file_index: FileIndex,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> std::io::Result<Self> {
        match file_index {
            FileIndex::Invalid => Ok(EagerLazyItemSet::new()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                ..
            } => {
                if offset == u32::MAX {
                    return Ok(EagerLazyItemSet::new());
                }
                reader.seek(SeekFrom::Start(offset as u64))?;
                let mut items = Vec::new();
                let mut current_chunk = offset;
                loop {
                    for i in 0..CHUNK_SIZE {
                        reader.seek(SeekFrom::Start(current_chunk as u64 + (i as u64 * 8)))?;
                        let item_offset = reader.read_u32::<LittleEndian>()?;
                        let version = reader.read_u32::<LittleEndian>()?.into();
                        if item_offset == u32::MAX {
                            continue;
                        }
                        let item_file_index = FileIndex::Valid {
                            offset: FileOffset(item_offset),
                            version,
                        };
                        let item = EagerLazyItem::deserialize(
                            reader,
                            item_file_index,
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?;
                        items.push(item);
                    }
                    reader.seek(SeekFrom::Start(
                        current_chunk as u64 + CHUNK_SIZE as u64 * 8,
                    ))?;
                    // Read next chunk link
                    current_chunk = reader.read_u32::<LittleEndian>()?;
                    if current_chunk == u32::MAX {
                        break;
                    }
                }
                Ok(EagerLazyItemSet::from_set(IdentitySet::from_iter(
                    items.into_iter(),
                )))
            }
        }
    }
}
