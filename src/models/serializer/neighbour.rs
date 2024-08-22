use super::CustomSerialize;
use crate::distance::cosine::CosineSimilarity;
use crate::models::types::FileOffset;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{FileIndex, LazyItem},
    types::Neighbour,
};
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
        writer.write_f32::<LittleEndian>(self.cosine_similarity.0)?;
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
        file_index: FileIndex,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> std::io::Result<Self> {
        match file_index {
            FileIndex::Invalid => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot deserialize Neighbour with an invalid FileIndex",
            )),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version,
            } => {
                reader.seek(SeekFrom::Start(offset as u64))?;
                // Deserialize the node position
                let node_pos = reader.read_u32::<LittleEndian>()?;
                // Deserialize the cosine similarity
                let cosine_similarity = reader.read_f32::<LittleEndian>()?;
                // Deserialize the node using the node position
                let node_file_index = FileIndex::Valid {
                    offset: FileOffset(node_pos),
                    version,
                };
                let node = LazyItem::deserialize(reader, node_file_index, cache, max_loads, skipm)?;
                Ok(Neighbour {
                    node,
                    cosine_similarity: CosineSimilarity(cosine_similarity),
                })
            }
        }
    }
}
