mod eager_lazy_item;
mod eager_lazy_item_set;
mod lazy_item;
mod lazy_item_map;
mod lazy_item_set;
mod metric_distance;
mod neighbour;
mod node;
mod vector;

#[cfg(test)]
mod tests;

use super::cache_loader::NodeRegistry;
use super::lazy_load::FileIndex;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

pub trait CustomSerialize {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32>;
    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        file_index: FileIndex,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> std::io::Result<Self>
    where
        Self: Sized;
}

impl CustomSerialize for f32 {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let pos = writer.stream_position()? as u32;
        writer.write_f32::<LittleEndian>(*self)?;
        Ok(pos)
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
            FileIndex::Valid { offset, .. } => {
                reader.seek(SeekFrom::Start(offset.0 as u64))?;
                reader.read_f32::<LittleEndian>()
            }
            FileIndex::Invalid => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot deserialize f32 with an invalid FileIndex",
            )),
        }
    }
}
