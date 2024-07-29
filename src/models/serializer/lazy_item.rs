use super::CustomSerialize;
use crate::models::{
    cache_loader::NodeRegistry,
    chunked_list::{LazyItem, LazyItemRef},
    types::{MergedNode, Neighbour},
};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, RwLock},
};

impl CustomSerialize for LazyItem<MergedNode> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        if let Some(existing_offset) = self.offset {
            if let Some(data) = &self.data {
                let guard = data.read().unwrap();
                if guard.needs_persistence() {
                    writer.seek(SeekFrom::Start(existing_offset as u64))?;
                    guard.set_persistence(false);
                    guard.serialize(writer)?;
                }
                Ok(existing_offset)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Attempting to serialize LazyItem with no data",
                ))
            }
        } else {
            if let Some(data) = &self.data {
                data.read().unwrap().serialize(writer)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Attempting to serialize LazyItem with no data",
                ))
            }
        }
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
    ) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let item = cache.get_object(offset, reader, MergedNode::deserialize, max_loads)?;

        Ok(item)
    }
}

impl CustomSerialize for LazyItem<Neighbour> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        if let Some(existing_offset) = self.offset {
            writer.seek(SeekFrom::Start(existing_offset as u64))?;
            if let Some(data) = &self.data {
                data.read().unwrap().serialize(writer)?;
                Ok(existing_offset)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Attempting to serialize LazyItem with no data",
                ))
            }
        } else {
            if let Some(data) = &self.data {
                data.read().unwrap().serialize(writer)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Attempting to serialize LazyItem with no data",
                ))
            }
        }
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
    ) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let data = Neighbour::deserialize(reader, offset, cache, max_loads)?;

        Ok(LazyItem {
            data: Some(Arc::new(RwLock::new(data))),
            offset: Some(offset),
            decay_counter: 0,
        })
    }
}

impl CustomSerialize for LazyItemRef<MergedNode> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let lazy_item = self.item.read().unwrap();
        let offset = if lazy_item.offset.is_none() {
            drop(lazy_item);
            self.set_offset(Some(writer.stream_position()? as u32));
            let lazy_item = self.item.read().unwrap();
            lazy_item.serialize(writer)?
        } else {
            lazy_item.serialize(writer)?
        };

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: u32,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
    ) -> std::io::Result<Self> {
        let lazy = LazyItem::deserialize(reader, offset, cache, max_loads)?;

        Ok(LazyItemRef {
            item: Arc::new(RwLock::new(lazy)),
        })
    }
}
