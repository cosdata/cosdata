use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    sync::Arc,
};

use dashmap::DashMap;

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::{Cacheable, NodeRegistry},
        lazy_load::{FileIndex, LazyItemArray},
        types::FileOffset,
        versioning::Hash,
    },
    storage::inverted_index_old::InvertedIndexItem,
};

use super::CustomSerialize;

impl<T> CustomSerialize for InvertedIndexItem<T>
where
    T: Cacheable + CustomSerialize + Clone + 'static,
    InvertedIndexItem<T>: Cacheable,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start_pos = bufman.cursor_position(cursor)? as u32;
        bufman.write_u32_with_cursor(cursor, self.dim_index)?;
        bufman.write_u8_with_cursor(cursor, if self.implicit { 1 } else { 0 })?;
        let placeholder_pos = bufman.cursor_position(cursor)?;
        bufman.write_u32_with_cursor(cursor, u32::MAX)?;
        bufman.write_u32_with_cursor(cursor, u32::MAX)?;
        let data_offset = self.data.serialize(bufmans.clone(), version, cursor)?;
        let children_offset = self.lazy_children.serialize(bufmans, version, cursor)?;
        let current_pos = bufman.cursor_position(cursor)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
        bufman.write_u32_with_cursor(cursor, data_offset)?;
        bufman.write_u32_with_cursor(cursor, children_offset)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(current_pos))?;
        Ok(start_pos)
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
                "Cannot deserialize MergedNode with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                offset: FileOffset(offset),
                version_number,
                version_id,
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                let dim_index = bufman.read_u32_with_cursor(cursor)?;
                let implicit = bufman.read_u8_with_cursor(cursor)? != 0;
                let data_offset = bufman.read_u32_with_cursor(cursor)?;
                let children_offset = bufman.read_u32_with_cursor(cursor)?;

                let data_file_index = FileIndex::Valid {
                    offset: FileOffset(data_offset),
                    version_number,
                    version_id,
                };
                let data = Arc::new(DashMap::deserialize(
                    bufmans.clone(),
                    data_file_index,
                    cache.clone(),
                    max_loads,
                    skipm,
                )?);

                let children_file_index = FileIndex::Valid {
                    offset: FileOffset(children_offset),
                    version_number,
                    version_id,
                };
                let lazy_children = LazyItemArray::deserialize(
                    bufmans,
                    children_file_index,
                    cache,
                    max_loads,
                    skipm,
                )?;

                Ok(Self {
                    dim_index,
                    implicit,
                    data,
                    lazy_children,
                })
            }
        }
    }
}
