use super::CustomSerialize;
use crate::distance::cosine::CosineSimilarity;
use crate::models::buffered_io::{BufIoError, BufferManagerFactory};
use crate::models::types::FileOffset;
use crate::models::versioning::Hash;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{FileIndex, LazyItem},
    types::Neighbour,
};
use std::collections::HashSet;
use std::{
    io::{self, SeekFrom},
    sync::Arc,
};

impl CustomSerialize for Neighbour {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let offset = bufman.cursor_position(cursor)? as u32;

        // Serialize the node position placeholder
        let node_placeholder = bufman.cursor_position(cursor)?;
        bufman.write_u32_with_cursor(cursor, 0)?;

        // Serialize the cosine similarity
        bufman.write_f32_with_cursor(cursor, self.cosine_similarity.0)?;
        let node_pos = self.node.serialize(bufmans, version, cursor)?;

        let end_pos = bufman.cursor_position(cursor)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(node_placeholder))?;
        // Serialize actual node position
        bufman.write_u32_with_cursor(cursor, node_pos)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(end_pos))?;

        Ok(offset)
    }
    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize Neighbour with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_id,
                version_number,
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                // Deserialize the node position
                let node_pos = bufman.read_u32_with_cursor(cursor)?;
                // Deserialize the cosine similarity
                let cosine_similarity = bufman.read_f32_with_cursor(cursor)?;
                // Deserialize the node using the node position
                let node_file_index = FileIndex::Valid {
                    offset: FileOffset(node_pos),
                    version_id,
                    version_number,
                };
                bufman.close_cursor(cursor)?;
                let node =
                    LazyItem::deserialize(bufmans, node_file_index, cache, max_loads, skipm)?;
                Ok(Neighbour {
                    node,
                    cosine_similarity: CosineSimilarity(cosine_similarity),
                })
            }
        }
    }
}
