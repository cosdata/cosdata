use super::CustomSerialize;
use crate::models::chunked_list::{LazyItem, LazyItemRef};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, RwLock},
};

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItem<T> {
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

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        reader.seek(SeekFrom::Start(offset as u64))?;
        let item = T::deserialize(reader, offset)?;

        Ok(LazyItem {
            data: Some(Arc::new(RwLock::new(item))),
            offset: Some(offset),
            decay_counter: 0,
        })
    }
}

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItemRef<T> {
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

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        let lazy = LazyItem::deserialize(reader, offset)?;

        Ok(LazyItemRef {
            item: Arc::new(RwLock::new(lazy)),
        })
    }
}
