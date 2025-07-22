use std::ptr;

use rustc_hash::FxHashSet;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager, FilelessBufferManager},
        cache_loader::HNSWIndexCache,
        lazy_item::FileIndex,
        prob_node::{LatestNode, SharedLatestNode, SharedNode},
        types::FileOffset,
    },
};

use super::{HNSWIndexSerialize, RawDeserialize};

impl HNSWIndexSerialize for SharedLatestNode {
    fn serialize(
        &self,
        bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        cursor: u64,
        latest_version_links_cursor: u64,
    ) -> Result<u32, BufIoError> {
        if self.is_null() {
            return Ok(u32::MAX);
        }
        let this = unsafe { &**self };
        let file_index = unsafe { &*this.latest }.file_index;
        latest_version_links_bufman
            .seek_with_cursor(latest_version_links_cursor, this.file_offset.0 as u64)?;
        latest_version_links_bufman
            .update_u32_with_cursor(latest_version_links_cursor, file_index.offset.0)?;
        latest_version_links_bufman
            .update_u32_with_cursor(latest_version_links_cursor, *file_index.file_id)?;
        this.latest.serialize(
            bufman,
            latest_version_links_bufman,
            cursor,
            latest_version_links_cursor,
        )?;
        Ok(this.file_offset.0)
    }

    fn deserialize(
        bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        file_index: FileIndex<IndexFileId>,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<Self, BufIoError> {
        if file_index.offset.0 == u32::MAX {
            return Ok(ptr::null_mut());
        }
        let cursor = latest_version_links_bufman.open_cursor()?;
        latest_version_links_bufman.seek_with_cursor(cursor, file_index.offset.0 as u64)?;
        let offset = FileOffset(latest_version_links_bufman.read_u32_with_cursor(cursor)?);
        let file_id = IndexFileId::from(latest_version_links_bufman.read_u32_with_cursor(cursor)?);
        latest_version_links_bufman.close_cursor(cursor)?;
        let item_file_index = FileIndex { offset, file_id };
        let latest = SharedNode::deserialize(
            bufman,
            latest_version_links_bufman,
            item_file_index,
            cache,
            max_loads,
            skipm,
        )?;
        Ok(LatestNode::new(latest, file_index.offset))
    }
}

impl RawDeserialize for SharedLatestNode {
    type Raw = FileIndex<IndexFileId>;

    fn deserialize_raw(
        _bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        _cursor: u64,
        latest_version_links_cursor: u64,
        offset: FileOffset,
        _file_id: IndexFileId,
        _cache: &HNSWIndexCache,
    ) -> Result<Self::Raw, BufIoError> {
        if offset.0 == u32::MAX {
            return Ok(FileIndex {
                offset: FileOffset(u32::MAX),
                file_id: IndexFileId::invalid(),
            });
        }
        latest_version_links_bufman
            .seek_with_cursor(latest_version_links_cursor, offset.0 as u64)?;

        let item_offset = FileOffset(
            latest_version_links_bufman.read_u32_with_cursor(latest_version_links_cursor)?,
        );
        let item_file_id = IndexFileId::from(
            latest_version_links_bufman.read_u32_with_cursor(latest_version_links_cursor)?,
        );
        let file_index = FileIndex {
            offset: item_offset,
            file_id: item_file_id,
        };

        Ok(file_index)
    }
}
