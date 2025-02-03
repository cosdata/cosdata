use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::{FileIndex, SyncPersist},
    prob_node::SharedNode,
    types::{FileOffset, MetricResult},
    versioning::Hash,
};

use super::ProbSerialize;

// @SERIALIZED_SIZE:
//   2 bytes for length +
//   length * (
//     4 bytes for id +
//     10 bytes offset & version +
//     5 bytes for distance/similarity
//   ) = 2 + len * 19
impl ProbSerialize for Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        level_0_bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
        direct: bool,
        is_level_0: bool,
    ) -> Result<u32, BufIoError> {
        let bufman = if is_level_0 {
            level_0_bufmans.get(version)?
        } else {
            bufmans.get(version)?
        };
        let start = bufman.cursor_position(cursor)?;
        let placeholder_start = start + 2;
        bufman.write_u16_with_cursor(cursor, self.len() as u16)?;

        if direct {
            for (i, neighbor) in self.iter().enumerate() {
                let (node_id, node_ptr, dist) = unsafe {
                    if let Some(neighbor) = neighbor.load(Ordering::SeqCst).as_ref() {
                        neighbor.clone()
                    } else {
                        continue;
                    }
                };

                let placeholder_pos = placeholder_start + (i as u64 * 19);

                let node_offset = node_ptr.serialize(
                    bufmans,
                    level_0_bufmans,
                    version,
                    cursor,
                    false,
                    is_level_0,
                )?;

                bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;

                let node = unsafe { &*node_ptr };

                bufman.write_u32_with_cursor(cursor, node_id)?;
                bufman.write_u32_with_cursor(cursor, node_offset)?;
                bufman.write_u16_with_cursor(cursor, node.get_current_version_number())?;
                bufman.write_u32_with_cursor(cursor, *node.get_current_version())?;
                crate::models::serializer::SimpleSerialize::serialize(&dist, &bufman, cursor)?;
            }
        }
        Ok(start as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        level_0_bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize neighbors with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                version_id,
                version_number: _,
                offset: FileOffset(offset),
            } => {
                let bufman = if is_level_0 {
                    level_0_bufmans.get(version_id)?
                } else {
                    bufmans.get(version_id)?
                };
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                let len = bufman.read_u16_with_cursor(cursor)? as usize;
                let mut neighbors = Vec::with_capacity(len);
                let placeholder_start = offset as u64 + 2;

                for i in 0..len {
                    let placeholder_offset = placeholder_start as u64 + i as u64 * 19;
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_offset))?;
                    let node_id = bufman.read_u32_with_cursor(cursor)?;
                    let node_offset = bufman.read_u32_with_cursor(cursor)?;
                    if node_offset == u32::MAX {
                        neighbors.push(AtomicPtr::new(ptr::null_mut()));
                        continue;
                    }
                    let node_version_number = bufman.read_u16_with_cursor(cursor)?;
                    let node_version_id = bufman.read_u32_with_cursor(cursor)?;

                    let dist: MetricResult =
                        crate::models::serializer::SimpleSerialize::deserialize(
                            &bufman,
                            FileOffset(placeholder_offset as u32 + 14),
                        )?;

                    let node_file_index = FileIndex::Valid {
                        offset: FileOffset(node_offset),
                        version_number: node_version_number,
                        version_id: Hash::from(node_version_id),
                    };

                    let node = SharedNode::deserialize(
                        bufmans,
                        level_0_bufmans,
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
    }
}
