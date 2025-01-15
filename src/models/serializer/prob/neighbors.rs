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

use super::{ProbSerialize, UpdateSerialized};

impl ProbSerialize for Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;

        let start_offset = bufman.cursor_position(cursor)?;
        bufman.write_u32_with_cursor(cursor, self.len() as u32)?;
        // (4 bytes for id + 10 bytes for node offset + 4 bytes for distance offset) * neighbors count
        bufman.write_with_cursor(cursor, &vec![u8::MAX; 18 * self.len()])?;

        let placeholder_start = start_offset + 4;

        for (i, neighbor) in self.iter().enumerate() {
            let (node_id, node_ptr, dist) = unsafe {
                if let Some(neighbor) = neighbor.load(Ordering::SeqCst).as_ref() {
                    neighbor.clone()
                } else {
                    continue;
                }
            };

            let placeholder_pos = placeholder_start + (i as u64 * 18);

            let node_offset = node_ptr.serialize(bufmans, version, cursor)?;
            let dist_offset = dist.serialize(bufmans, version, cursor)?;
            let end_offset = bufman.cursor_position(cursor)?;

            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;

            let node = unsafe { &*node_ptr };

            bufman.write_u32_with_cursor(cursor, node_id)?;
            bufman.write_u32_with_cursor(cursor, node_offset)?;
            bufman.write_u16_with_cursor(cursor, node.get_current_version_number())?;
            bufman.write_u32_with_cursor(cursor, *node.get_current_version())?;
            bufman.write_u32_with_cursor(cursor, dist_offset)?;

            bufman.seek_with_cursor(cursor, SeekFrom::Start(end_offset))?;
        }

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize neighbors with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                version_id,
                version_number,
                offset: FileOffset(offset),
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

                let len = bufman.read_u32_with_cursor(cursor)? as usize;
                let mut neighbors = Vec::with_capacity(len);

                let placeholder_start = offset as u64 + 4;

                for _ in 0..len {
                    neighbors.push(AtomicPtr::new(ptr::null_mut()));
                }

                for i in 0..len {
                    let placeholder_offset = placeholder_start as u64 + i as u64 * 18;
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_offset))?;
                    let node_id = bufman.read_u32_with_cursor(cursor)?;
                    let node_offset = bufman.read_u32_with_cursor(cursor)?;
                    if node_offset == u32::MAX {
                        continue;
                    }
                    let node_version_number = bufman.read_u16_with_cursor(cursor)?;
                    let node_version_id = bufman.read_u32_with_cursor(cursor)?;
                    let dist_offset = bufman.read_u32_with_cursor(cursor)?;

                    let node_file_index = FileIndex::Valid {
                        offset: FileOffset(node_offset),
                        version_number: node_version_number,
                        version_id: Hash::from(node_version_id),
                    };

                    let dist_file_index = FileIndex::Valid {
                        offset: FileOffset(dist_offset),
                        version_number,
                        version_id,
                    };

                    let node =
                        SharedNode::deserialize(bufmans, node_file_index, cache, max_loads, skipm)?;

                    let dist = MetricResult::deserialize(
                        bufmans,
                        dist_file_index,
                        cache,
                        max_loads,
                        skipm,
                    )?;

                    let ptr = Box::into_raw(Box::new((node_id, node, dist)));

                    neighbors[i].store(ptr, Ordering::SeqCst);
                }

                bufman.close_cursor(cursor)?;

                Ok(neighbors.into_boxed_slice())
            }
        }
    }
}

impl UpdateSerialized for Box<[AtomicPtr<(u32, SharedNode, MetricResult)>]> {
    fn update_serialized(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
    ) -> Result<u32, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot update neighbors with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                version_id,
                offset: FileOffset(offset),
                ..
            } => {
                let bufman = bufmans.get(version_id)?;
                let cursor = bufman.open_cursor()?;
                let placeholder_offset = offset as u64 + 4;

                for i in 0..self.len() {
                    let (node_id, node_ptr, dist) = unsafe {
                        if let Some(neighbor) = self[i].load(Ordering::SeqCst).as_ref() {
                            neighbor.clone()
                        } else {
                            continue;
                        }
                    };

                    bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

                    let placeholder_pos = placeholder_offset + (i as u64 * 18);

                    let node_offset = node_ptr.serialize(bufmans, version_id, cursor)?;
                    let dist_offset = dist.serialize(bufmans, version_id, cursor)?;
                    let end_offset = bufman.cursor_position(cursor)?;

                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;

                    let node = unsafe { &*node_ptr };

                    bufman.write_u32_with_cursor(cursor, node_id)?;
                    bufman.write_u32_with_cursor(cursor, node_offset)?;
                    bufman.write_u16_with_cursor(cursor, node.get_current_version_number())?;
                    bufman.write_u32_with_cursor(cursor, *node.get_current_version())?;
                    bufman.write_u32_with_cursor(cursor, dist_offset)?;

                    bufman.seek_with_cursor(cursor, SeekFrom::Start(end_offset))?;
                }
                bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

                Ok(offset)
            }
        }
    }
}
