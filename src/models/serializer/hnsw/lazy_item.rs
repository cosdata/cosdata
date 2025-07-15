use rustc_hash::FxHashSet;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager, FilelessBufferManager},
        cache_loader::HNSWIndexCache,
        lazy_item::FileIndex,
        prob_node::SharedNode,
    },
};

use super::HNSWIndexSerialize;

impl HNSWIndexSerialize for SharedNode {
    fn serialize(
        &self,
        bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        cursor: u64,
        latest_version_links_cursor: u64,
    ) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        let file_offset = lazy_item.file_index.offset.0;

        if let Some(data) = lazy_item.unsafe_get_data() {
            bufman.seek_with_cursor(cursor, file_offset as u64)?;
            data.serialize(
                bufman,
                latest_version_links_bufman,
                cursor,
                latest_version_links_cursor,
            )?;
        }

        Ok(file_offset)
    }

    fn deserialize(
        _bufman: &BufferManager,
        _latest_version_links_bufman: &FilelessBufferManager,
        file_index: FileIndex<IndexFileId>,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<Self, BufIoError> {
        cache.get_lazy_object(file_index, max_loads, skipm)
    }
}
