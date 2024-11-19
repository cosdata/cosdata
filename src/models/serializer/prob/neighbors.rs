use std::{
    collections::HashSet,
    io::{self, SeekFrom},
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::{FileIndex, SyncPersist},
    prob_lazy_load::lazy_item::ProbLazyItem,
    prob_node::{ProbNode, SharedNode},
    types::{FileOffset, MetricResult},
    versioning::Hash,
};

use super::{ProbSerialize, UpdateSerialized};

impl<const N: usize> ProbSerialize for [AtomicPtr<(SharedNode, MetricResult)>; N] {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(&version)?;

        let start_offset = bufman.cursor_position(cursor)?;
        // (10 bytes for node offset + 4 bytes for distance offset) * neighbors count
        bufman.write_with_cursor(cursor, &vec![u8::MAX; 14 * N])?;

        for i in 0..N {
            let (node, dist) = unsafe {
                if let Some(neighbor) = self[i].load(Ordering::SeqCst).as_ref() {
                    neighbor.clone()
                } else {
                    continue;
                }
            };

            let placeholder_pos = start_offset + (i as u64 * 14);

            let node_offset = node.serialize(bufmans.clone(), version, cursor)?;
            let dist_offset = dist.serialize(bufmans.clone(), version, cursor)?;
            let end_offset = bufman.cursor_position(cursor)?;

            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;

            bufman.write_u32_with_cursor(cursor, node_offset)?;
            bufman.write_u16_with_cursor(cursor, node.get_current_version_number())?;
            bufman.write_u32_with_cursor(cursor, *node.get_current_version())?;
            bufman.write_u32_with_cursor(cursor, dist_offset)?;

            bufman.seek_with_cursor(cursor, SeekFrom::Start(end_offset))?;
        }

        Ok(start_offset as u32)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<ProbCache>,
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
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;

                let neighbors = std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut()));

                for i in 0..N {
                    let placeholder_offset = offset as u64 + i as u64 * 14;
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_offset))?;
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

                    let node = Arc::<ProbLazyItem<ProbNode>>::deserialize(
                        bufmans.clone(),
                        node_file_index,
                        cache.clone(),
                        max_loads,
                        skipm,
                    )?;

                    let dist = MetricResult::deserialize(
                        bufmans.clone(),
                        dist_file_index,
                        cache.clone(),
                        max_loads,
                        skipm,
                    )?;

                    let ptr = Box::into_raw(Box::new((node, dist)));

                    neighbors[i].store(ptr, Ordering::SeqCst);
                }

                bufman.close_cursor(cursor)?;

                Ok(neighbors)
            }
        }
    }
}

impl<const N: usize> UpdateSerialized for [AtomicPtr<(SharedNode, MetricResult)>; N] {
    fn update_serialized(
        &self,
        bufmans: Arc<BufferManagerFactory>,
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
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;
                let start_offset = offset as u64;

                for i in 0..N {
                    let (node, dist) = unsafe {
                        if let Some(neighbor) = self[i].load(Ordering::SeqCst).as_ref() {
                            neighbor.clone()
                        } else {
                            continue;
                        }
                    };

                    bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;

                    let placeholder_pos = start_offset + (i as u64 * 14);

                    let node_offset = node.serialize(bufmans.clone(), version_id, cursor)?;
                    let dist_offset = dist.serialize(bufmans.clone(), version_id, cursor)?;
                    let end_offset = bufman.cursor_position(cursor)?;

                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;

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
