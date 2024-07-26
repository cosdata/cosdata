use super::CustomSerialize;
use crate::models::{
    chunked_list::SyncPersist,
    types::{MergedNode, Neighbour},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

impl CustomSerialize for Neighbour {
    fn serialize<W: Write + Seek>(&mut self, writer: &mut W) -> std::io::Result<u32> {
        if !self.needs_persistence() {
            return Ok(u32::MAX);
        }

        self.set_persistence(false);

        let offset = writer.stream_position()? as u32;

        // Serialize the node
        Arc::make_mut(&mut self.node).serialize(writer)?;

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
