mod eager_lazy_item;
mod eager_lazy_item_set;
mod lazy_item;
mod lazy_item_map;
mod lazy_item_set;
mod neighbour;
mod node;
mod vector;

#[cfg(test)]
mod tests;

use super::cache_loader::NodeRegistry;
use crate::models::types::FileOffset;
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
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<FileOffset>,
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
        offset: u32,
        _cache: Arc<NodeRegistry<R>>,
        _max_loads: u16,
        _skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        reader.seek(SeekFrom::Start(offset as u64))?;
        reader.read_f32::<LittleEndian>()
    }
}
