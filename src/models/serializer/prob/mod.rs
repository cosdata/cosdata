mod lazy_item;
mod lazy_item_array;
mod metric_distance;
mod neighbors;
mod node;
#[cfg(test)]
mod tests;

pub use lazy_item::lazy_item_deserialize_impl;

use std::{collections::HashSet, sync::Arc};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::ProbCache,
    lazy_load::FileIndex,
    versioning::Hash,
};

pub trait ProbSerialize: Sized {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<ProbCache>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError>;
}

pub trait UpdateSerialized {
    fn update_serialized(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
    ) -> Result<u32, BufIoError>;
}
