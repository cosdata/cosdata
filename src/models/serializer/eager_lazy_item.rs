use super::CustomSerialize;
use crate::models::{
    cache_loader::NodeRegistry,
    lazy_load::{EagerLazyItem, FileIndex, LazyItem, SyncPersist},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

impl<T, E> CustomSerialize for EagerLazyItem<T, E>
where
    T: Clone + 'static,
    LazyItem<T>: CustomSerialize,
    E: Clone + CustomSerialize + 'static,
{
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let start = writer.stream_position()? as u32;
        self.0.serialize(writer)?;
        let item_placeholder = writer.stream_position()?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u16::<LittleEndian>(self.1.get_current_version())?;
        let item_offset = self.1.serialize(writer)?;
        let end_position = writer.stream_position()?;

        writer.seek(SeekFrom::Start(item_placeholder))?;
        writer.write_u32::<LittleEndian>(item_offset)?;
        writer.seek(SeekFrom::Start(end_position))?;

        Ok(start)
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        file_index: FileIndex,
        cache: Arc<NodeRegistry<R>>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        match file_index {
            FileIndex::Invalid => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot deserialize EagerLazyItem with an invalid FileIndex",
            )),
            FileIndex::Valid { offset, .. } => {
                reader.seek(SeekFrom::Start(offset as u64))?;
                let eager_data =
                    E::deserialize(reader, file_index, cache.clone(), max_loads, skipm)?;
                let item_offset = reader.read_u32::<LittleEndian>()?;
                let version = reader.read_u16::<LittleEndian>()?;
                let item_file_index = FileIndex::Valid {
                    offset: item_offset,
                    version,
                };
                let item = LazyItem::deserialize(reader, item_file_index, cache, max_loads, skipm)?;
                Ok(Self(eager_data, item))
            }
        }
    }
}
