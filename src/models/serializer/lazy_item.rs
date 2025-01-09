use super::CustomSerialize;
use crate::models::buffered_io::{BufIoError, BufferManagerFactory};
use crate::models::cache_loader::Cacheable;
use crate::models::lazy_load::LazyItemVec;
use crate::models::lazy_load::SyncPersist;
use crate::models::lazy_load::{FileIndex, CHUNK_SIZE};
use crate::models::types::FileOffset;
use crate::models::versioning::Hash;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{LazyItem, LazyItemRef},
    types::MergedNode,
};
use arcshift::ArcShift;
use std::collections::HashSet;
use std::{
    io::{self, SeekFrom},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

fn update_map<T>(
    bufmans: Arc<BufferManagerFactory<Hash>>,
    version: Hash,
    cursor: u64,
    offset: u32,
    map: &LazyItemVec<T>,
) -> Result<u32, BufIoError>
where
    T: Clone + CustomSerialize + Cacheable + 'static,
{
    let bufman = bufmans.get(version)?;
    let offset = if offset == u32::MAX {
        // Serialize a new map
        bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
        map.serialize(bufmans, version, cursor)?
    } else {
        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
        let items = map.items.clone().get().clone();

        let mut current_chunk = offset;

        let mut i = 0;

        // remove the items from `items` which have already been serialized
        let last_chunk = loop {
            for j in 0..CHUNK_SIZE {
                bufman.seek_with_cursor(
                    cursor,
                    SeekFrom::Start(current_chunk as u64 + (j as u64 * 10)),
                )?;
                let item_offset = bufman.read_u32_with_cursor(cursor)?;
                if item_offset == u32::MAX {
                    continue;
                }
                if let Some(item) = items.get(i) {
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(item_offset as u64))?;
                    item.serialize(bufmans.clone(), version, cursor)?;
                }
                i += 1;
            }
            let prev_chunk = current_chunk;
            bufman.seek_with_cursor(
                cursor,
                SeekFrom::Start(current_chunk as u64 + CHUNK_SIZE as u64 * 10),
            )?;
            current_chunk = bufman.read_u32_with_cursor(cursor)?;
            if current_chunk == u32::MAX {
                break prev_chunk;
            }
        };

        bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
        // fill last chunk
        for j in 0..CHUNK_SIZE {
            if i >= items.len() {
                break;
            }
            let current_pos = bufman.cursor_position(cursor)?;
            let placeholder_pos = last_chunk as u64 + (j as u64 * 10);
            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
            let item_offset = bufman.read_u32_with_cursor(cursor)?;
            if item_offset != u32::MAX {
                bufman.seek_with_cursor(cursor, SeekFrom::Start(current_pos))?;
                continue;
            }
            let item = &items[i];
            bufman.seek_with_cursor(cursor, SeekFrom::Start(current_pos))?;
            let item_offset = item.serialize(bufmans.clone(), version, cursor)?;

            let current_pos = bufman.cursor_position(cursor)?;
            bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
            bufman.write_u32_with_cursor(cursor, item_offset)?;
            bufman.write_u16_with_cursor(cursor, item.get_current_version_number())?;
            bufman.write_u32_with_cursor(cursor, *item.get_current_version())?;

            bufman.seek_with_cursor(cursor, SeekFrom::Start(current_pos))?;
            i += 1;
        }

        if items.len() > i {
            let total_items = items.len() - i;

            let current_pos = bufman.cursor_position(cursor)?;
            bufman.seek_with_cursor(
                cursor,
                SeekFrom::Start(last_chunk as u64 + (CHUNK_SIZE as u64 * 10)),
            )?;
            bufman.write_u32_with_cursor(cursor, current_pos as u32)?;
            bufman.seek_with_cursor(cursor, SeekFrom::Start(current_pos))?;

            for chunk_start in (0..total_items).step_by(CHUNK_SIZE) {
                let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE, total_items);
                let is_last_chunk = chunk_end == total_items;

                // Write placeholders for item offsets
                let placeholder_start = bufman.cursor_position(cursor)? as u32;
                for _ in 0..CHUNK_SIZE {
                    bufman.write_u32_with_cursor(cursor, u32::MAX)?;
                    bufman.write_u16_with_cursor(cursor, u16::MAX)?;
                    bufman.write_u32_with_cursor(cursor, u32::MAX)?;
                }
                // Write placeholder for next chunk link
                let next_chunk_placeholder = bufman.cursor_position(cursor)? as u32;
                bufman.write_u32_with_cursor(cursor, u32::MAX)?;

                // Serialize items and update placeholders
                for j in chunk_start..chunk_end {
                    let item_offset = items[i].serialize(bufmans.clone(), version, cursor)?;
                    let placeholder_pos =
                        placeholder_start as u64 + ((j - chunk_start) as u64 * 10);
                    let current_pos = bufman.cursor_position(cursor)?;

                    // Write entry offset
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(placeholder_pos))?;
                    bufman.write_u32_with_cursor(cursor, item_offset)?;
                    bufman.write_u16_with_cursor(cursor, items[i].get_current_version_number())?;
                    bufman.write_u32_with_cursor(cursor, *items[i].get_current_version())?;

                    // Return to the current position
                    bufman.seek_with_cursor(cursor, SeekFrom::Start(current_pos))?;
                    i += 1;
                }

                // Write next chunk link
                let next_chunk_start = bufman.cursor_position(cursor)? as u32;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(next_chunk_placeholder as u64))?;
                if is_last_chunk {
                    bufman.write_u32_with_cursor(cursor, u32::MAX)?; // Last chunk
                } else {
                    bufman.write_u32_with_cursor(cursor, next_chunk_start)?;
                }
                bufman.seek_with_cursor(cursor, SeekFrom::Start(next_chunk_start as u64))?;
            }
        }

        offset
    };

    Ok(offset)
}

fn lazy_item_serialize_impl<T: Cacheable + CustomSerialize>(
    node: Arc<T>,
    versions: &LazyItemVec<T>,
    bufmans: Arc<BufferManagerFactory<Hash>>,
    version: Hash,
    cursor: u64,
    serialized_flag: bool,
) -> Result<u32, BufIoError> {
    let bufman = bufmans.get(version)?;
    let node_placeholder = bufman.cursor_position(cursor)?;
    if serialized_flag {
        let _node_offset = bufman.read_u32_with_cursor(cursor)?;
        let versions_placeholder = bufman.cursor_position(cursor)?;
        let versions_offset = bufman.read_u32_with_cursor(cursor)?;
        let offset = update_map(bufmans, version, cursor, versions_offset, versions)?;
        let current_position = bufman.cursor_position(cursor)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(versions_placeholder))?;
        bufman.write_u32_with_cursor(cursor, offset)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(current_position))?;
    } else {
        bufman.write_u32_with_cursor(cursor, 0)?;
        let versions_placeholder = bufman.cursor_position(cursor)?;
        bufman.write_u32_with_cursor(cursor, 0)?;
        let node_offset = node.serialize(bufmans.clone(), version, cursor)?;
        let versions_offset = versions.serialize(bufmans.clone(), version, cursor)?;
        let end_offset = bufman.cursor_position(cursor)?;

        bufman.seek_with_cursor(cursor, SeekFrom::Start(node_placeholder))?;
        bufman.write_u32_with_cursor(cursor, node_offset)?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(versions_placeholder))?;
        bufman.write_u32_with_cursor(cursor, versions_offset)?;

        bufman.seek_with_cursor(cursor, SeekFrom::Start(end_offset))?;
    }

    Ok(node_placeholder as u32)
}

fn lazy_item_deserialize_impl<T: Cacheable + CustomSerialize + Clone>(
    bufmans: Arc<BufferManagerFactory<Hash>>,
    file_index: FileIndex,
    cache: Arc<NodeRegistry>,
    max_loads: u16,
    skipm: &mut HashSet<u64>,
) -> Result<LazyItem<T>, BufIoError> {
    match file_index {
        FileIndex::Invalid => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Cannot deserialize MergedNode with an invalid FileIndex",
        )
        .into()),
        FileIndex::Valid {
            offset,
            version_id,
            version_number,
        } => {
            if offset.0 == u32::MAX {
                return Ok(LazyItem::Invalid);
            }
            let bufman = bufmans.get(version_id)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, SeekFrom::Start(offset.0 as u64))?;
            let node_offset = bufman.read_u32_with_cursor(cursor)?;
            let versions_offset = bufman.read_u32_with_cursor(cursor)?;
            let data = T::deserialize(
                bufmans.clone(),
                FileIndex::Valid {
                    offset: FileOffset(node_offset),
                    version_id,
                    version_number,
                },
                cache.clone(),
                max_loads,
                skipm,
            )?;
            let versions = LazyItemVec::deserialize(
                bufmans.clone(),
                FileIndex::Valid {
                    offset: FileOffset(versions_offset),
                    version_number,
                    version_id,
                },
                cache,
                max_loads,
                skipm,
            )?;

            Ok(LazyItem::Valid {
                data: ArcShift::new(Some(Arc::new(data))),
                file_index: ArcShift::new(Some(file_index)),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions,
                version_id,
                version_number,
                serialized_flag: Arc::new(AtomicBool::new(true)),
            })
        }
    }
}

impl<T: Cacheable + CustomSerialize> CustomSerialize for LazyItem<T> {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        match self {
            Self::Valid {
                data,
                file_index,
                versions,
                version_id,
                serialized_flag: serialized_flag_arc,
                ..
            } => {
                let mut data_arc = data.clone();
                let bufman = bufmans.get(*version_id)?;
                if let Some(existing_file_index) = file_index.clone().get().clone() {
                    if let FileIndex::Valid {
                        offset: FileOffset(offset),
                        ..
                    } = existing_file_index
                    {
                        if let Some(data) = data_arc.get() {
                            if self.needs_persistence() {
                                let cursor = if version_id == &version {
                                    cursor
                                } else {
                                    bufman.open_cursor()?
                                };
                                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                                self.set_persistence(false);
                                let serialized_flag =
                                    serialized_flag_arc.swap(true, Ordering::Relaxed);
                                lazy_item_serialize_impl(
                                    data.clone(),
                                    versions,
                                    bufmans,
                                    *version_id,
                                    cursor,
                                    serialized_flag,
                                )?;
                                if version_id != &version {
                                    bufman.close_cursor(cursor)?;
                                }
                            }
                        }
                        return Ok(offset);
                    }
                }

                if let Some(data) = data_arc.get() {
                    let cursor = if version_id == &version {
                        cursor
                    } else {
                        let cursor = bufman.open_cursor()?;
                        bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
                        cursor
                    };
                    let offset = bufman.cursor_position(cursor)? as u32;
                    let item_version_id = self.get_current_version();
                    let item_version_number = self.get_current_version_number();

                    self.set_file_index(Some(FileIndex::Valid {
                        offset: FileOffset(offset),
                        version_id: item_version_id,
                        version_number: item_version_number,
                    }));

                    self.set_persistence(false);
                    let serialized_flag = serialized_flag_arc.swap(true, Ordering::Relaxed);
                    let offset = lazy_item_serialize_impl(
                        data.clone(),
                        versions,
                        bufmans,
                        *version_id,
                        cursor,
                        serialized_flag,
                    )?;
                    if version_id != &version {
                        bufman.close_cursor(cursor)?;
                    }
                    Ok(offset)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Attempting to serialize LazyItem with no data",
                    )
                    .into())
                }
            }
            Self::Invalid => Ok(u32::MAX),
        }
    }

    fn deserialize(
        _reader: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        cache.get_object(file_index, lazy_item_deserialize_impl, max_loads, skipm)
    }
}

impl CustomSerialize for LazyItemRef<MergedNode> {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let mut arc = self.item.clone();
        let lazy_item = arc.get();
        let offset = lazy_item.serialize(bufmans, version, cursor)?;
        Ok(offset)
    }

    fn deserialize(
        reader: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let lazy = LazyItem::deserialize(reader, file_index, cache, max_loads, skipm)?;
        Ok(LazyItemRef {
            item: ArcShift::new(lazy),
        })
    }
}
