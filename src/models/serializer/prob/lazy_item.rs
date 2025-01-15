use std::{collections::HashSet, io::SeekFrom, sync::atomic::Ordering};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::FileIndex,
    prob_lazy_load::lazy_item::{ProbLazyItemState, ReadyState},
    prob_node::SharedNode,
    types::FileOffset,
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

impl ProbSerialize for SharedNode {
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
                persist_flag,
                version_id,
                version_number,
                ..
            }) => {
                let bufman = bufmans.get(*version_id)?;

                let offset = if let Some(file_offset) = file_offset.get() {
                    if !persist_flag.swap(false, Ordering::SeqCst) {
                        return Ok(file_offset.0);
                    }

                    let cursor = if version_id == &version {
                        cursor
                    } else {
                        bufman.open_cursor()?
                    };

                    bufman.seek_with_cursor(cursor, SeekFrom::Start(file_offset.0 as u64))?;

                    let file_index = FileIndex::Valid {
                        offset: file_offset,
                        version_number: *version_number,
                        version_id: *version_id,
                    };
                    data.update_serialized(bufmans, file_index)?;

                    if version_id != &version {
                        bufman.close_cursor(cursor)?;
                    }

                    file_offset.0
                } else {
                    let cursor = if version_id == &version {
                        cursor
                    } else {
                        bufman.open_cursor()?
                    };

                    let offset = bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

                    file_offset.set(Some(FileOffset(u32::try_from(offset).unwrap())));
                    persist_flag.store(false, Ordering::SeqCst);

                    data.serialize(bufmans, *version_id, cursor)?;

                    if version_id != &version {
                        bufman.close_cursor(cursor)?;
                    }

                    offset as u32
                };

                Ok(offset)
            }
        }
    }

    fn deserialize(
        _bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        cache.get_lazy_object(file_index, max_loads, skipm)
    }
}
