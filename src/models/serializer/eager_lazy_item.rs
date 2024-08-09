use super::CustomSerialize;
use crate::models::{
    cache_loader::NodeRegistry,
    chunked_list::{EagerLazyItem, LazyItem},
    types::FileOffset,
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
        let item_offset = self.1.serialize(writer)?;
        let end_position = writer.stream_position()?;

        writer.seek(SeekFrom::Start(item_placeholder))?;
        writer.write_u32::<LittleEndian>(item_offset)?;
        writer.seek(SeekFrom::Start(end_position))?;

        Ok(start)
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
        let eager_data = E::deserialize(reader, offset, cache.clone(), max_loads, skipm)?;
        let item_offset = reader.read_u32::<LittleEndian>()?;
        let item = LazyItem::deserialize(reader, item_offset, cache, max_loads, skipm)?;

        Ok(Self(eager_data, item))
    }
}
