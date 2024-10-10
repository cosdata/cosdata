use crate::{
    distance::{
        cosine::{CosineDistance, CosineSimilarity},
        dotproduct::DotProductDistance,
        euclidean::EuclideanDistance,
        hamming::HammingDistance,
    },
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::NodeRegistry,
        lazy_load::FileIndex,
        types::MetricResult,
        versioning::Hash,
    },
};
use std::collections::HashSet;
use std::{
    io::{self, SeekFrom},
    sync::Arc,
};

use super::CustomSerialize;

impl CustomSerialize for MetricResult {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(&version)?;
        let (variant, value) = match self {
            Self::CosineSimilarity(value) => (0, value.0),
            Self::CosineDistance(value) => (1, value.0),
            Self::EuclideanDistance(value) => (2, value.0),
            Self::HammingDistance(value) => (3, value.0),
            Self::DotProductDistance(value) => (4, value.0),
        };
        let start = bufman.cursor_position(cursor)? as u32;
        bufman.write_u8_with_cursor(cursor, variant)?;
        bufman.write_f32_with_cursor(cursor, value)?;
        Ok(start)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        _cache: Arc<NodeRegistry>,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => {
                let bufman = bufmans.get(&version_id)?;
                let cursor = bufman.open_cursor()?;
                bufman.seek_with_cursor(cursor, SeekFrom::Start(offset.0 as u64))?;
                let variant = bufman.read_u8_with_cursor(cursor)?;
                let value = bufman.read_f32_with_cursor(cursor)?;
                let metric = match variant {
                    0 => Self::CosineSimilarity(CosineSimilarity(value)),
                    1 => Self::CosineDistance(CosineDistance(value)),
                    2 => Self::EuclideanDistance(EuclideanDistance(value)),
                    3 => Self::HammingDistance(HammingDistance(value)),
                    4 => Self::DotProductDistance(DotProductDistance(value)),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Invalid MetricResult variant",
                        )
                        .into());
                    }
                };
                bufman.close_cursor(cursor)?;
                Ok(metric)
            }
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize f32 with an invalid FileIndex",
            )
            .into()),
        }
    }
}
