use super::CustomSerialize;
use crate::models::lazy_load::FileIndex;
use crate::models::lazy_load::SyncPersist;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{LazyItem, LazyItemRef},
    types::MergedNode,
};
use arcshift::ArcShift;
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

impl CustomSerialize for LazyItem<MergedNode> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        match self {
            Self::Valid {
                data, file_index, ..
            } => {
                if let Some(existing_file_index) = file_index.clone().get().clone() {
                    if let FileIndex::Valid { offset, .. } = existing_file_index {
                        if let Some(data) = &data {
                            let mut arc = data.clone();
                            let data = arc.get();
                            if self.needs_persistence() {
                                writer.seek(SeekFrom::Start(offset as u64))?;
                                self.set_persistence(false);
                                data.serialize(writer)?;
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
                    let offset = data.serialize(writer)?;
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
        match file_index {
            FileIndex::Valid { offset, version } => {
                let combined_index = NodeRegistry::<R>::combine_index(&file_index);
                reader.seek(SeekFrom::Start(offset as u64))?;
                let item = cache.get_object(
                    file_index,
                    reader,
                    MergedNode::deserialize,
                    max_loads,
                    skipm,
                )?;
                Ok(item)
            }
            FileIndex::Invalid => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )),
        }
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
