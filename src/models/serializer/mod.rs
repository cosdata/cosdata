mod cuckoo_filter_tree;
mod dashmap;
mod eager_lazy_item;
mod eager_lazy_item_set;
mod incremental_serializable_growable_data;
mod inverted_index;
mod inverted_index_item;
mod inverted_index_sparse_ann_node;
mod lazy_item;
mod lazy_item_array;
mod lazy_item_map;
mod lazy_item_set;
mod lazy_item_vec;
mod metric_distance;
mod neighbour;
mod node;
pub mod prob;
mod storage;

#[cfg(test)]
mod tests;

use super::buffered_io::{BufIoError, BufferManager, BufferManagerFactory};
use super::cache_loader::NodeRegistry;
use super::lazy_load::FileIndex;
use super::types::FileOffset;
use super::versioning::Hash;
use std::collections::HashSet;
use std::io::{self, SeekFrom};
use std::sync::Arc;

pub trait CustomSerialize: Sized {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>;
}

trait SimpleSerialize: Sized {
    fn serialize(&self, bufman: Arc<BufferManager>, cursor: u64) -> Result<u32, BufIoError>;

    fn deserialize(bufman: Arc<BufferManager>, offset: FileOffset) -> Result<Self, BufIoError>;
}

impl<T: SimpleSerialize> CustomSerialize for T {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        SimpleSerialize::serialize(self, bufman, cursor)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        _cache: Arc<NodeRegistry>,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => {
                let bufman = bufmans.get(version_id)?;
                SimpleSerialize::deserialize(bufman, offset)
            }
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )
            .into()),
        }
    }
}

impl SimpleSerialize for f32 {
    fn serialize(&self, bufman: Arc<BufferManager>, cursor: u64) -> Result<u32, BufIoError> {
        let offset = bufman.cursor_position(cursor)? as u32;
        bufman.write_f32_with_cursor(cursor, *self)?;
        Ok(offset)
    }

    fn deserialize(
        bufman: Arc<BufferManager>,
        FileOffset(offset): FileOffset,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
        let res = bufman.read_f32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(res)
    }
}

impl SimpleSerialize for u32 {
    fn serialize(&self, bufman: Arc<BufferManager>, cursor: u64) -> Result<u32, BufIoError> {
        let offset = bufman.cursor_position(cursor)? as u32;
        bufman.write_u32_with_cursor(cursor, *self)?;
        Ok(offset)
    }

    fn deserialize(
        bufman: Arc<BufferManager>,
        FileOffset(offset): FileOffset,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
        let res = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(res)
    }
}
