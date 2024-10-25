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
mod storage;

#[cfg(test)]
mod tests;

use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::cache_loader::NodeRegistry;
use super::lazy_load::FileIndex;
use super::versioning::Hash;
use std::collections::HashSet;
use std::io::{self, SeekFrom};
use std::sync::Arc;

pub trait CustomSerialize: Sized {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>;
}

impl CustomSerialize for f32 {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(&version)?;
        let offset = bufman.cursor_position(cursor)? as u32;
        bufman.write_f32_with_cursor(cursor, *self)?;
        Ok(offset)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        _cache: Arc<NodeRegistry>,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => {
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset.0 as u64))?;
                let res = bufman.read_f32_with_cursor(cursor)?;
                bufman.close_cursor(cursor)?;
                Ok(res)
            }
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize f32 with an invalid FileIndex",
            )
            .into()),
        }
    }
}

impl CustomSerialize for u32 {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(&version)?;
        let offset = bufman.cursor_position(cursor)? as u32;
        bufman.write_u32_with_cursor(cursor, *self)?;
        Ok(offset)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        _cache: Arc<NodeRegistry>,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => {
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset.0 as u64))?;
                let res = bufman.read_u32_with_cursor(cursor)?;
                bufman.close_cursor(cursor)?;
                Ok(res)
            }
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize u32 with an invalid FileIndex",
            )
            .into()),
        }
    }
}
