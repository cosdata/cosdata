use super::CustomSerialize;
use crate::models::lazy_load::FileIndex;
use crate::models::lazy_load::LazyItemMap;
use crate::models::lazy_load::SyncPersist;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{LazyItem, LazyItemRef},
    types::MergedNode,
};
use arcshift::ArcShift;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{atomic::AtomicBool, Arc},
};

fn lazy_item_serialize_impl<W: Write + Seek>(
    node: &MergedNode,
    versions: &LazyItemMap<MergedNode>,
    writer: &mut W,
) -> std::io::Result<u32> {
    let node_placeholder = writer.stream_position()?;
    writer.write_u32::<LittleEndian>(0)?;
    let versions_placeholder = writer.stream_position()?;
    writer.write_u32::<LittleEndian>(0)?;
    let node_offset = node.serialize(writer)?;
    let versions_offset = versions.serialize(writer)?;
    let end_offset = writer.stream_position()?;

    writer.seek(SeekFrom::Start(node_placeholder))?;
    writer.write_u32::<LittleEndian>(node_offset)?;
    writer.seek(SeekFrom::Start(versions_placeholder))?;
    writer.write_u32::<LittleEndian>(versions_offset)?;

    writer.seek(SeekFrom::Start(end_offset))?;

    Ok(node_placeholder as u32)
}

fn lazy_item_deserialize_impl<R: Read + Seek>(
    reader: &mut R,
    file_index: FileIndex,
    cache: Arc<NodeRegistry<R>>,
    max_loads: u16,
    skipm: &mut HashSet<u64>,
) -> std::io::Result<LazyItem<MergedNode>> {
    match file_index {
        FileIndex::Invalid => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Cannot deserialize MergedNode with an invalid FileIndex",
        )),
        FileIndex::Valid { offset, version } => {
            if offset == u32::MAX {
                return Ok(LazyItem::Invalid);
            }
            reader.seek(SeekFrom::Start(offset as u64))?;
            let node_offset = reader.read_u32::<LittleEndian>()?;
            let versions_offset = reader.read_u32::<LittleEndian>()?;
            let data = MergedNode::deserialize(
                reader,
                FileIndex::Valid {
                    offset: node_offset,
                    version,
                },
                cache.clone(),
                max_loads,
                skipm,
            )?;
            let versions = LazyItemMap::deserialize(
                reader,
                FileIndex::Valid {
                    offset: versions_offset,
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
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        match self {
            Self::Valid {
                data,
                file_index,
                versions,
                ..
            } => {
                if let Some(existing_file_index) = file_index.clone().get().clone() {
                    if let FileIndex::Valid { offset, .. } = existing_file_index {
                        if let Some(data) = &data {
                            let mut arc = data.clone();
                            let data = arc.get();
                            if self.needs_persistence() {
                                writer.seek(SeekFrom::Start(offset as u64))?;
                                self.set_persistence(false);
                                lazy_item_serialize_impl(data, versions, writer)?;
                            }
                            return Ok(offset);
                        }
                    }
                }

                if let Some(data) = &data {
                    let mut arc = data.clone();
                    let offset = writer.stream_position()? as u32;
                    let version = self.get_current_version();
                    self.set_file_index(Some(FileIndex::Valid { offset, version }));

                    let data = arc.get();
                    self.set_persistence(false);
                    let offset = lazy_item_serialize_impl(data, versions, writer)?;
                    Ok(offset)
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Attempting to serialize LazyItem with no data",
                    ))
                }
            }
            Self::Invalid => Ok(u32::MAX),
        }
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        file_index: FileIndex,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> std::io::Result<Self> {
        cache.get_object(
            file_index,
            reader,
            lazy_item_deserialize_impl,
            max_loads,
            skipm,
        )
    }
}

impl CustomSerialize for LazyItemRef<MergedNode> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let mut arc = self.item.clone();
        let lazy_item = arc.get();
        let offset = lazy_item.serialize(writer)?;
        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        file_index: FileIndex,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> std::io::Result<Self> {
        let lazy = LazyItem::deserialize(reader, file_index, cache, max_loads, skipm)?;
        Ok(LazyItemRef {
            item: ArcShift::new(lazy),
        })
    }
}
