use std::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use rustc_hash::FxHashMap;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager},
        cache_loader::HNSWIndexCache,
        prob_lazy_load::lazy_item::{FileIndex, ProbLazyItem},
        prob_node::SharedNode,
        serializer::SimpleSerialize,
        types::{FileOffset, InternalId, MetricResult},
    },
};

use super::HNSWIndexSerialize;

// @SERIALIZED_SIZE:
//   2 bytes for length +
//   length * (
//     4 bytes for id +
//     8 bytes offset & file id +
//     5 bytes for distance/similarity
//   ) = 2 + len * 17
impl HNSWIndexSerialize for Box<[AtomicPtr<(InternalId, SharedNode, MetricResult)>]> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let start = bufman.cursor_position(cursor)?;
        bufman.update_u16_with_cursor(cursor, self.len() as u16)?;

        for neighbor in self.iter() {
            let (node_id, node_ptr, dist) = unsafe {
                if let Some(neighbor) = neighbor.load(Ordering::SeqCst).as_ref() {
                    *neighbor
                } else {
                    bufman.update_with_cursor(cursor, &[u8::MAX; 17])?;
                    continue;
                }
            };

            let node = unsafe { &*node_ptr };

            let FileIndex {
                offset: node_offset,
                file_id: node_file_id,
            } = node.file_index;
            let mut buf = Vec::with_capacity(17);
            buf.extend(node_id.to_le_bytes());
            buf.extend(node_offset.0.to_le_bytes());
            buf.extend(node_file_id.to_le_bytes());
            let (tag, value) = dist.get_tag_and_value();
            buf.push(tag);
            buf.extend(value.to_le_bytes());
            bufman.update_with_cursor(cursor, &buf)?;
        }
        Ok(start as u32)
    }

    fn deserialize(
        bufman: &BufferManager,
        FileOffset(offset): FileOffset,
        _file_id: IndexFileId,
        cache: &HNSWIndexCache,
        pending_items: &mut FxHashMap<FileIndex, SharedNode>,
    ) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        let len = bufman.read_u16_with_cursor(cursor)? as usize;
        let mut neighbors = Vec::with_capacity(len);
        let placeholder_start = offset as u64 + 2;

        for i in 0..len {
            let placeholder_offset = placeholder_start + i as u64 * 17;
            bufman.seek_with_cursor(cursor, placeholder_offset)?;
            let node_id = InternalId::from(bufman.read_u32_with_cursor(cursor)?);
            let node_offset = bufman.read_u32_with_cursor(cursor)?;
            if node_offset == u32::MAX {
                neighbors.push(AtomicPtr::new(ptr::null_mut()));
                continue;
            }
            let node_file_id = bufman.read_u32_with_cursor(cursor)?;

            let dist: MetricResult =
                SimpleSerialize::deserialize(bufman, FileOffset(placeholder_offset as u32 + 12))?;

            let node_file_index = FileIndex {
                offset: FileOffset(node_offset),
                file_id: IndexFileId::from(node_file_id),
            };

            let node = cache
                .registry
                .get(&HNSWIndexCache::combine_index(&node_file_index))
                .unwrap_or_else(|| {
                    *pending_items
                        .entry(node_file_index)
                        .or_insert_with(|| ProbLazyItem::new_pending(node_file_index))
                });
            let ptr = Box::into_raw(Box::new((node_id, node, dist)));

            neighbors.push(AtomicPtr::new(ptr));
        }

        Ok(neighbors.into_boxed_slice())
    }
}
