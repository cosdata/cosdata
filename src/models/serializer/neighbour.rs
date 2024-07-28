use super::CustomSerialize;
use crate::models::{
    cache_loader::NodeRegistry,
    chunked_list::LazyItem,
    types::{MergedNode, Neighbour},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, RwLock},
};

impl CustomSerialize for Neighbour {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        // Serialize the node
        self.node.read().unwrap().serialize(writer)?;

        // Serialize the cosine similarity
        writer.write_f32::<LittleEndian>(self.cosine_similarity)?;

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
    ) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        // Deserialize the node
        let node = MergedNode::deserialize(reader, offset, cache, max_loads)?;

        // Deserialize the cosine similarity
        let cosine_similarity = reader.read_f32::<LittleEndian>()?;

        Ok(Neighbour {
            node: Arc::new(RwLock::new(node)),
            cosine_similarity,
        })
    }
}
