mod lazy_item;
mod lazy_item_array;
mod neighbors;
mod node;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::DenseIndexCache,
    lazy_load::FileIndex,
    types::FileOffset,
    versioning::Hash,
};

pub trait DenseSerialize: Sized {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &DenseIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError>;
}

pub trait DenseUpdateSerialized {
    fn update_serialized(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        offset: FileOffset,
        cursor: u64,
    ) -> Result<u32, BufIoError>;
}
