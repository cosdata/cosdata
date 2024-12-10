mod lazy_item;
mod lazy_item_array;
mod neighbors;
mod node;
#[cfg(test)]
mod tests;

use std::{collections::HashSet, io};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::FileIndex,
    versioning::Hash,
};

use super::SimpleSerialize;

pub trait ProbSerialize: Sized {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>;
}

pub trait UpdateSerialized {
    fn update_serialized(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
    ) -> Result<u32, BufIoError>;
}

impl<T: SimpleSerialize> ProbSerialize for T {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = bufmans.get(version)?;
        SimpleSerialize::serialize(self, bufman, cursor)
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        _cache: &ProbCache,
        _max_loads: u16,
        _skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        match file_index {
            FileIndex::Invalid => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )
            .into()),
            FileIndex::Valid {
                version_id, offset, ..
            } => {
                let bufman = bufmans.get(version_id)?;
                SimpleSerialize::deserialize(bufman, offset)
            }
        }
    }
}
