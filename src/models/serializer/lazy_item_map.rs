use super::CustomSerialize;
use crate::models::identity_collections::{IdentityMap, IdentityMapKey};
use crate::models::lazy_load::{FileIndex, LazyItemMap, SyncPersist};
use crate::models::types::FileOffset;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{LazyItem, CHUNK_SIZE},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

const MSB: u32 = 1 << 31;

impl<T> CustomSerialize for LazyItemMap<T>
where
    T: Clone + CustomSerialize + 'static,
    LazyItem<T>: CustomSerialize,
{
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        if self.is_empty() {
            return Ok(u32::MAX);
        };
        let start_offset = writer.stream_position()? as u32;
        let mut items_arc = self.items.clone();
        let items: Vec<_> = items_arc
            .get()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
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
                let entry_offset = items[i].0.serialize(writer)?;
                let item_placeholder_pos = writer.stream_position()?;
                writer.write_u32::<LittleEndian>(0)?;
                let item_offset = items[i].1.serialize(writer)?;

                let placeholder_pos = placeholder_start as u64 + ((i - chunk_start) as u64 * 8);
                let current_pos = writer.stream_position()?;

                // Write entry offset
                writer.seek(SeekFrom::Start(placeholder_pos))?;
                writer.write_u32::<LittleEndian>(entry_offset)?;
                writer.write_u32::<LittleEndian>(*items[i].1.get_current_version())?;

                // Write item offset
                writer.seek(SeekFrom::Start(item_placeholder_pos))?;
                writer.write_u32::<LittleEndian>(item_offset)?;

                // Return to the current position
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
            FileIndex::Invalid => Ok(LazyItemMap::new()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                ..
            } => {
                if offset == u32::MAX {
                    return Ok(LazyItemMap::new());
                }
                reader.seek(SeekFrom::Start(offset as u64))?;
                let mut items = Vec::new();
                let mut current_chunk = offset;
                loop {
                    for i in 0..CHUNK_SIZE {
                        reader.seek(SeekFrom::Start(current_chunk as u64 + (i as u64 * 8)))?;
                        let entry_offset = reader.read_u32::<LittleEndian>()?;
                        let version = reader.read_u32::<LittleEndian>()?.into();
                        if entry_offset == u32::MAX {
                            continue;
                        }
                        let entry_file_index = FileIndex::Valid {
                            offset: FileOffset(entry_offset),
                            version,
                        };
                        let key = IdentityMapKey::deserialize(
                            reader,
                            entry_file_index,
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?;
                        let item_offset = reader.read_u32::<LittleEndian>()?;
                        let item_file_index = FileIndex::Valid {
                            offset: FileOffset(item_offset),
                            version,
                        };
                        let item = LazyItem::deserialize(
                            reader,
                            item_file_index,
                            cache.clone(),
                            max_loads,
                            skipm,
                        )?;
                        items.push((key, item));
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
                Ok(LazyItemMap::from_map(IdentityMap::from_iter(
                    items.into_iter(),
                )))
            }
        }
    }
}

impl CustomSerialize for IdentityMapKey {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let start = writer.stream_position()? as u32;
        match self {
            Self::String(str) => {
                let bytes = str.clone().into_bytes();
                let len = bytes.len() as u32;
                writer.write_u32::<LittleEndian>(MSB | len)?;
                writer.write_all(&bytes)?;
            }
            Self::Int(int) => {
                writer.write_u32::<LittleEndian>(*int)?;
            }
        }
        Ok(start)
    }
    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        file_index: FileIndex,
        _cache: Arc<NodeRegistry<R>>,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        match file_index {
            FileIndex::Valid {
                offset: FileOffset(offset),
                ..
            } => {
                reader.seek(SeekFrom::Start(offset as u64))?;
                let num = reader.read_u32::<LittleEndian>()?;
                if num & MSB == 0 {
                    return Ok(Self::Int(num));
                }
                let len = (num << 1) >> 1;
                let mut bytes = vec![0; len as usize];
                reader.read_exact(&mut bytes)?;
                let str = String::from_utf8(bytes).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid identity map key: {}", e),
                    )
                })?;
                Ok(Self::String(str))
            }
            FileIndex::Invalid => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot deserialize IdentityMapKey with an invalid FileIndex",
            )),
        }
    }
}
