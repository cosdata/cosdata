use std::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use rustc_hash::FxHashSet;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager, FilelessBufferManager},
        cache_loader::HNSWIndexCache,
        lazy_item::FileIndex,
        prob_node::{Neighbors, SharedLatestNode},
        serializer::SimpleSerialize,
        types::{FileOffset, InternalId, MetricResult},
    },
};

use super::{HNSWIndexSerialize, RawDeserialize};

// @SERIALIZED_SIZE:
//   2 bytes for length +
//   length * (
//     4 bytes for id +
//     4 bytes for offset +
//     5 bytes for distance/similarity
//   ) = 2 + len * 13
impl HNSWIndexSerialize for Neighbors {
    fn serialize(
        &self,
        bufman: &BufferManager,
        _latest_version_links_bufman: &FilelessBufferManager,
        cursor: u64,
        _latest_version_links_cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start = bufman.cursor_position(cursor)?;
        bufman.update_u16_with_cursor(cursor, self.len() as u16)?;

        for neighbor in self.iter() {
            let (node_id, node_ptr, dist) = unsafe {
                if let Some(neighbor) = neighbor.load(Ordering::SeqCst).as_ref() {
                    *neighbor
                } else {
                    bufman.update_with_cursor(cursor, &[u8::MAX; 13])?;
                    continue;
                }
            };

            let node_offset = unsafe { &*node_ptr }.file_offset;

            let mut buf = Vec::with_capacity(13);
            buf.extend(node_id.to_le_bytes());
            buf.extend(node_offset.0.to_le_bytes());
            let (tag, value) = dist.get_tag_and_value();
            buf.push(tag);
            buf.extend(value.to_le_bytes());
            bufman.update_with_cursor(cursor, &buf)?;
        }
        Ok(start as u32)
    }

    fn deserialize(
        bufman: &BufferManager,
        latest_version_links_bufman: &FilelessBufferManager,
        file_index: FileIndex<IndexFileId>,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        let offset = file_index.offset.0;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        let len = bufman.read_u16_with_cursor(cursor)? as usize;
        let mut neighbors = Vec::with_capacity(len);
        let placeholder_start = offset as u64 + 2;

        for i in 0..len {
            let placeholder_offset = placeholder_start + i as u64 * 13;
            bufman.seek_with_cursor(cursor, placeholder_offset)?;
            let node_id = InternalId::from(bufman.read_u32_with_cursor(cursor)?);
            let node_offset = bufman.read_u32_with_cursor(cursor)?;
            if node_offset == u32::MAX {
                neighbors.push(AtomicPtr::new(ptr::null_mut()));
                continue;
            }

            let dist: MetricResult =
                SimpleSerialize::deserialize(bufman, FileOffset(placeholder_offset as u32 + 8))?;

            let node_file_index = FileIndex {
                offset: FileOffset(node_offset),
                file_id: IndexFileId::invalid(),
            };
            let node = SharedLatestNode::deserialize(
                bufman,
                latest_version_links_bufman,
                node_file_index,
                cache,
                max_loads,
                skipm,
            )?;
            let ptr = Box::into_raw(Box::new((node_id, node, dist)));

            neighbors.push(AtomicPtr::new(ptr));
        }

        bufman.close_cursor(cursor)?;

        Ok(neighbors.into_boxed_slice())
    }
}

impl RawDeserialize for Neighbors {
    type Raw = Vec<Option<(InternalId, FileOffset, MetricResult)>>;

    fn deserialize_raw(
        bufman: &BufferManager,
        _latest_version_links_bufman: &FilelessBufferManager,
        cursor: u64,
        _latest_version_links_cursor: u64,
        FileOffset(offset): FileOffset,
        _file_id: IndexFileId,
        _cache: &HNSWIndexCache,
    ) -> Result<Self::Raw, BufIoError> {
        bufman.seek_with_cursor(cursor, offset as u64)?;
        let len = bufman.read_u16_with_cursor(cursor)? as usize;
        let mut neighbors = Vec::with_capacity(len);
        let placeholder_start = offset as u64 + 2;

        for i in 0..len {
            let placeholder_offset = placeholder_start + i as u64 * 13;
            bufman.seek_with_cursor(cursor, placeholder_offset)?;
            let node_id = InternalId::from(bufman.read_u32_with_cursor(cursor)?);
            let node_offset = bufman.read_u32_with_cursor(cursor)?;
            if node_offset == u32::MAX {
                neighbors.push(None);
                continue;
            }

            let dist: MetricResult =
                SimpleSerialize::deserialize(bufman, FileOffset(placeholder_offset as u32 + 8))?;

            neighbors.push(Some((node_id, FileOffset(node_offset), dist)));
        }

        Ok(neighbors)
    }
}
