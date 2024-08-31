use super::CustomSerialize;
use crate::models::buffered_io::{BufIoError, BufferManagerFactory};
use crate::models::lazy_load::FileIndex;
use crate::models::lazy_load::LazyItemMap;
use crate::models::lazy_load::SyncPersist;
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
    sync::{atomic::AtomicBool, Arc},
};

fn lazy_item_serialize_impl(
    node: &MergedNode,
    versions: &LazyItemMap<MergedNode>,
    bufmans: Arc<BufferManagerFactory>,
    version: Hash,
    cursor: u64,
) -> Result<u32, BufIoError> {
    let bufman = bufmans.get(&version)?;
    let node_placeholder = bufman.cursor_position(cursor)?;
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

    Ok(node_placeholder as u32)
}

fn lazy_item_deserialize_impl(
    bufmans: Arc<BufferManagerFactory>,
    file_index: FileIndex,
    cache: Arc<NodeRegistry>,
    max_loads: u16,
    skipm: &mut HashSet<u64>,
) -> Result<LazyItem<MergedNode>, BufIoError> {
    match file_index {
        FileIndex::Invalid => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Cannot deserialize MergedNode with an invalid FileIndex",
        )
        .into()),
        FileIndex::Valid { offset, version } => {
            if offset.0 == u32::MAX {
                return Ok(LazyItem::Invalid);
            }
            let bufman = bufmans.get(&version)?;
            let cursor = bufman.open_cursor()?;
            bufman.seek_with_cursor(cursor, SeekFrom::Start(offset.0 as u64))?;
            let node_offset = bufman.read_u32_with_cursor(cursor)?;
            let versions_offset = bufman.read_u32_with_cursor(cursor)?;
            let data = MergedNode::deserialize(
                bufmans.clone(),
                FileIndex::Valid {
                    offset: FileOffset(node_offset),
                    version,
                },
                cache.clone(),
                max_loads,
                skipm,
            )?;
            let versions = LazyItemMap::deserialize(
                bufmans.clone(),
                FileIndex::Valid {
                    offset: FileOffset(versions_offset),
                    version,
                },
                cache,
                max_loads,
                skipm,
            )?;
            Ok(LazyItem::Valid {
                data: Some(ArcShift::new(data)),
                file_index: ArcShift::new(Some(file_index)),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions,
                version_id: version,
            })
        }
    }
}

impl CustomSerialize for LazyItem<MergedNode> {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        match self {
            Self::Valid {
                data,
                file_index,
                versions,
                version_id,
                ..
            } => {
                let bufman = bufmans.get(version_id)?;
                if let Some(existing_file_index) = file_index.clone().get().clone() {
                    if let FileIndex::Valid {
                        offset: FileOffset(offset),
                        ..
                    } = existing_file_index
                    {
                        if let Some(data) = &data {
                            let mut arc = data.clone();
                            let data = arc.get();
                            if self.needs_persistence() {
                                let cursor = if version_id == &version {
                                    cursor
                                } else {
                                    bufman.open_cursor()?
                                };
                                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;
                                self.set_persistence(false);
                                lazy_item_serialize_impl(
                                    data,
                                    versions,
                                    bufmans,
                                    *version_id,
                                    cursor,
                                )?;
                                if version_id != &version {
                                    bufman.close_cursor(cursor)?;
                                }
                            }
                            return Ok(offset);
                        }
                    }
                }

                if let Some(data) = &data {
                    let cursor = if version_id == &version {
                        cursor
                    } else {
                        let cursor = bufman.open_cursor()?;
                        bufman.seek_with_cursor(cursor, SeekFrom::End(0))?;
                        cursor
                    };
                    let mut arc = data.clone();
                    let offset = bufman.cursor_position(cursor)? as u32;
                    let version = self.get_current_version();

                    self.set_file_index(Some(FileIndex::Valid {
                        offset: FileOffset(offset),
                        version,
                    }));

                    let data = arc.get();
                    self.set_persistence(false);
                    let offset =
                        lazy_item_serialize_impl(data, versions, bufmans, *version_id, cursor)?;
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
        _reader: Arc<BufferManagerFactory>,
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
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let mut arc = self.item.clone();
        let lazy_item = arc.get();
        let offset = lazy_item.serialize(bufmans, version, cursor)?;
        Ok(offset)
    }

    fn deserialize(
        reader: Arc<BufferManagerFactory>,
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
