use super::CustomSerialize;
use crate::models::chunked_list::LazyItem;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

impl<T: Clone + CustomSerialize> CustomSerialize for LazyItem<T> {
    fn serialize<W: Write + Seek>(&mut self, writer: &mut W) -> std::io::Result<u32> {
        let offset = match self {
            LazyItem::Ready(item, offset) => {
                if let Some(existing_offset) = *offset {
                    writer.seek(SeekFrom::Start(existing_offset as u64))?;
                    Arc::make_mut(item).serialize(writer)?;
                    existing_offset
                } else {
                    let offs = Arc::make_mut(item).serialize(writer)?;
                    *offset = Some(offs);
                    offs
                }
            }
            LazyItem::LazyLoad(file_offset) => *file_offset,
            LazyItem::Null => u32::MAX,
        };

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        let item = T::deserialize(reader, offset)?;
        Ok(LazyItem::Ready(Arc::new(item), Some(offset)))
    }
}
