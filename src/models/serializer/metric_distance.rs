use crate::{
    distance::cosine::CosineSimilarity,
    models::{
        cache_loader::NodeRegistry,
        types::{FileOffset, MetricResult},
    },
};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashSet;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    sync::Arc,
};

use super::CustomSerialize;

impl CustomSerialize for MetricResult {
    fn serialize<W: Write + Seek>(&self, writer: &mut W) -> std::io::Result<u32> {
        let value = self.get_value();
        value.serialize(writer)
    }

    fn deserialize<R: Read + Seek>(
        reader: &mut R,
        offset: FileOffset,
        _cache: Arc<NodeRegistry<R>>,
        _max_loads: u16,
        _skipm: &mut HashSet<FileOffset>,
    ) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        reader.seek(SeekFrom::Start(offset.0 as u64))?;
        let value = reader.read_f32::<LittleEndian>()?;
        Ok(MetricResult::CosineSimilarity(CosineSimilarity(value)))
    }
}
