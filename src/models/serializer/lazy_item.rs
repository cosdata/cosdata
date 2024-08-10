use super::CustomSerialize;
use crate::models::lazy_load::SyncPersist;
use crate::models::types::{FileOffset, Item};
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{LazyItem, LazyItemRef},
    types::MergedNode,
};
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

impl CustomSerialize for LazyItem<MergedNode> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        match self {
            Self::Valid { data, offset, .. } => {
                if let Some(existing_offset) = offset.clone().get().clone() {
                    if let Some(data) = &data {
                        let mut arc = data.clone();
                        let data = arc.get();
                        if data.needs_persistence() {
                            writer.seek(SeekFrom::Start(existing_offset as u64))?;
                            data.set_persistence(false);
                            data.serialize(writer)?;
                        }
                        Ok(existing_offset)
                    } else {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Attempting to serialize LazyItem with no data",
                        ))
                    }
                } else {
                    if let Some(data) = &data {
                        let mut arc = data.clone();
                        let offset = writer.stream_position()? as u32;
                        self.set_offset(Some(offset));
                        let data = arc.get();
                        data.set_persistence(false);
                        let offset = data.serialize(writer)?;
                        Ok(offset)
                    } else {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Attempting to serialize LazyItem with no data",
                        ))
                    }
                }
            }
            Self::Invalid => Ok(u32::MAX),
        }
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let item = cache.get_object(offset, reader, MergedNode::deserialize, max_loads, skipm)?;

        Ok(item)
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
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self> {
        let lazy = LazyItem::deserialize(reader, offset, cache, max_loads, skipm)?;

        Ok(LazyItemRef {
            item: Item::new(lazy),
        })
    }
}
