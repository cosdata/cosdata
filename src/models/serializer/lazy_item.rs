use super::CustomSerialize;
use crate::models::chunked_list::{LazyItem, LazyItemRef};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, RwLock},
};
impl<T: Clone + CustomSerialize> CustomSerialize for LazyItemRef<T> {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let lazy_item = self.read().unwrap();
        match &*lazy_item {
            LazyItem::Valid { data, offset, .. } => {
                if let Some(existing_offset) = offset {
                    writer.seek(SeekFrom::Start(*existing_offset as u64))?;
                    if let Some(data) = data {
                        data.serialize(writer)?;
                    } else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Attempting to serialize LazyItem with no data",
                        ));
                    }
                    Ok(*existing_offset)
                } else {
                    let new_offset = if let Some(data) = data {
                        data.serialize(writer)?
                    } else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Attempting to serialize LazyItem with no data",
                        ));
                    };
                    self.write().unwrap().set_offset(Some(new_offset));
                    Ok(new_offset)
                }
            }
            LazyItem::Invalid => Ok(u32::MAX),
        }
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let item = T::deserialize(reader, offset)?;
        Ok(Arc::new(RwLock::new(LazyItem::Valid {
            data: Some(Arc::new(item)),
            offset: Some(offset),
            decay_counter: 0,
        })))
    }
}
