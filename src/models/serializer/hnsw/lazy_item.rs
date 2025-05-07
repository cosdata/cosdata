use rustc_hash::FxHashSet;

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    cache_loader::HNSWIndexCache,
    prob_lazy_load::lazy_item::FileIndex,
    prob_node::SharedNode,
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
        _bufman: &BufferManager,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<Self, BufIoError> {
        cache.get_lazy_object(file_index, max_loads, skipm)
    }
}
