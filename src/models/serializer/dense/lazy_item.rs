use std::collections::HashSet;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::DenseIndexCache,
    lazy_load::FileIndex,
    prob_lazy_load::lazy_item::{ProbLazyItemState, ReadyState},
    prob_node::SharedNode,
    versioning::Hash,
};

use super::DenseSerialize;

impl DenseSerialize for SharedNode {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        match lazy_item.unsafe_get_state() {
            ProbLazyItemState::Pending(file_index) => Ok(file_index.get_offset().unwrap().0),
            ProbLazyItemState::Ready(ReadyState {
                data,
                file_offset,
                version_id,
                ..
            }) => {
                let bufman = bufmans.get(*version_id)?;

                let cursor = if version_id == &version {
                    cursor
                } else {
                    bufman.open_cursor()?
                };

                bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;

                data.serialize(bufmans, *version_id, cursor)?;

                if version_id != &version {
                    bufman.close_cursor(cursor)?;
                }

                Ok(file_offset.0)
            }
        }
    }

    fn deserialize(
        _bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &DenseIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError> {
        cache.get_lazy_object(file_index, max_loads, skipm, is_level_0)
    }
}
