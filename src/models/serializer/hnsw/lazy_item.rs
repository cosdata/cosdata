use std::collections::HashSet;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::HNSWIndexCache,
    prob_lazy_load::lazy_item::FileIndex,
    prob_node::SharedNode,
    versioning::Hash,
};

use super::HNSWIndexSerialize;

impl HNSWIndexSerialize for SharedNode {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        let file_offset = lazy_item.file_index.offset.0;

        if let Some(data) = lazy_item.unsafe_get_data() {
            let version_id = lazy_item.file_index.version_id;

            let bufman = bufmans.get(version_id)?;

            let cursor = if version_id == version {
                cursor
            } else {
                bufman.open_cursor()?
            };

            bufman.seek_with_cursor(cursor, file_offset as u64)?;

            data.serialize(bufmans, version_id, cursor)?;

            if version_id != version {
                bufman.close_cursor(cursor)?;
            }
        }

        Ok(file_offset)
    }

    fn deserialize(
        _bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError> {
        cache.get_lazy_object(file_index, max_loads, skipm, is_level_0)
    }
}
