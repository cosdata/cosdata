use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    inverted_index::InvertedIndexNodeData,
    prob_lazy_load::lazy_item::{ProbLazyItem, ProbLazyItemState, ReadyState},
    types::FileOffset,
};

use super::InvertedIndexSerialize;

impl InvertedIndexSerialize for *mut ProbLazyItem<InvertedIndexNodeData> {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let lazy_item = unsafe { &**self };
        match lazy_item.unsafe_get_state() {
            ProbLazyItemState::Pending(file_index) => Ok(file_index.offset.0),
            ProbLazyItemState::Ready(ReadyState {
                data, file_offset, ..
            }) => {
                dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
                data.serialize(
                    dim_bufman,
                    data_bufmans,
                    data_file_idx,
                    data_file_parts,
                    cursor,
                )?;
                Ok(file_offset.0)
            }
        }
    }

    fn deserialize(
        _dim_bufmans: &BufferManager,
        _data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        _data_file_parts: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError> {
        cache.get_data(file_offset, data_file_idx)
    }
}
