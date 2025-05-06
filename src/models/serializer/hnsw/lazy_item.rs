use rustc_hash::FxHashMap;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager},
        cache_loader::HNSWIndexCache,
        prob_lazy_load::lazy_item::{FileIndex, ProbLazyItem},
        prob_node::{ProbNode, SharedNode},
        types::FileOffset,
    },
};

use super::HNSWIndexSerialize;

impl HNSWIndexSerialize for SharedNode {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        let file_offset = lazy_item.file_index.offset.0;

        if let Some(data) = lazy_item.unsafe_get_data() {
            bufman.seek_with_cursor(cursor, file_offset as u64)?;
            data.serialize(bufman, cursor)?;
        }

        Ok(file_offset)
    }

    fn deserialize(
        bufman: &BufferManager,
        offset: FileOffset,
        file_id: IndexFileId,
        cache: &HNSWIndexCache,
        ready_items: &FxHashMap<FileIndex, SharedNode>,
        pending_items: &mut FxHashMap<FileIndex, SharedNode>,
    ) -> Result<Self, BufIoError> {
        Ok(ProbLazyItem::new(
            ProbNode::deserialize(bufman, offset, file_id, cache, ready_items, pending_items)?,
            file_id,
            offset,
        ))
    }
}
