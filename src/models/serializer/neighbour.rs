use super::CustomSerialize;
use crate::models::types::FileOffset;
use crate::models::{cache_loader::NodeRegistry, lazy_load::LazyItem, types::Neighbour};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

impl CustomSerialize for Neighbour {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        // Serialize the node position placeholder
        let node_placeholder = writer.stream_position()?;
        writer.write_u32::<LittleEndian>(0)?;

        // Serialize the cosine similarity
        writer.write_f32::<LittleEndian>(self.cosine_similarity)?;
        let node_pos = self.node.serialize(writer)?;

        let end_pos = writer.stream_position()?;
        writer.seek(SeekFrom::Start(node_placeholder))?;
        // Serialize actual node position
        writer.write_u32::<LittleEndian>(node_pos)?;
        writer.seek(SeekFrom::Start(end_pos))?;

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        // Deserialize the node
        let node_pos = reader.read_u32::<LittleEndian>()?;

        // Deserialize the cosine similarity
        let cosine_similarity = reader.read_f32::<LittleEndian>()?;
        let node = LazyItem::deserialize(reader, node_pos, cache, max_loads, skipm)?;

        Ok(Neighbour {
            node,
            cosine_similarity,
        })
    }
}
