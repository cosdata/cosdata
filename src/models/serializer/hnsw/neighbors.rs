use std::{
    collections::HashSet,
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::HNSWIndexCache,
    prob_lazy_load::lazy_item::FileIndex,
    prob_node::SharedNode,
    serializer::SimpleSerialize,
    types::{FileOffset, MetricResult},
    versioning::Hash,
};

use super::HNSWIndexSerialize;

// @SERIALIZED_SIZE:
//   2 bytes for length +
//   length * (
//     4 bytes for id +
//     10 bytes offset & version +
//     5 bytes for distance/similarity
//   ) = 2 + len * 19
impl HNSWIndexSerialize for Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        let start = bufman.cursor_position(cursor)?;
        debug_assert_eq!(
            (start - 47) % (self.len() as u64 * 19 + 129),
            0,
            "offset: {}",
            start
        );
        bufman.update_u16_with_cursor(cursor, self.len() as u16)?;

        for neighbor in self.iter() {
            let (node_id, node_ptr, dist) = unsafe {
                if let Some(neighbor) = neighbor.load(Ordering::SeqCst).as_ref() {
                    *neighbor
                } else {
                    bufman.update_with_cursor(cursor, &[u8::MAX; 19])?;
                    continue;
                }
            };

            let node = unsafe { &*node_ptr };

            let FileIndex {
                offset: node_offset,
                version_number: node_version_number,
                version_id: node_version_id,
            } = node.get_file_index();
            let mut buf = Vec::with_capacity(19);
            buf.extend(node_id.to_le_bytes());
            buf.extend(node_offset.0.to_le_bytes());
            buf.extend(node_version_number.to_le_bytes());
            buf.extend(node_version_id.to_le_bytes());
            let (tag, value) = dist.get_tag_and_value();
            buf.push(tag);
            buf.extend(value.to_le_bytes());
            bufman.update_with_cursor(cursor, &buf)?;
        }
        Ok(start as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError> {
        let FileIndex {
            version_id,
            version_number: _,
            offset: FileOffset(offset),
        } = file_index;
        let bufman = bufmans.get(version_id)?;
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        let len = bufman.read_u16_with_cursor(cursor)? as usize;
        let mut neighbors = Vec::with_capacity(len);
        let placeholder_start = offset as u64 + 2;

        for i in 0..len {
            let placeholder_offset = placeholder_start + i as u64 * 19;
            bufman.seek_with_cursor(cursor, placeholder_offset)?;
            let node_id = bufman.read_u32_with_cursor(cursor)?;
            let node_offset = bufman.read_u32_with_cursor(cursor)?;
            if node_offset == u32::MAX {
                neighbors.push(AtomicPtr::new(ptr::null_mut()));
                continue;
            }
            let node_version_number = bufman.read_u16_with_cursor(cursor)?;
            let node_version_id = bufman.read_u32_with_cursor(cursor)?;

            let dist: MetricResult =
                SimpleSerialize::deserialize(&bufman, FileOffset(placeholder_offset as u32 + 14))?;

            let node_file_index = FileIndex {
                offset: FileOffset(node_offset),
                version_number: node_version_number,
                version_id: Hash::from(node_version_id),
            };

            let node = SharedNode::deserialize(
                bufmans,
                node_file_index,
                cache,
                max_loads,
                skipm,
                is_level_0,
            )?;
            let ptr = Box::into_raw(Box::new((node_id, node, dist)));

            neighbors.push(AtomicPtr::new(ptr));
        }

        Ok(neighbors.into_boxed_slice())
    }
}
