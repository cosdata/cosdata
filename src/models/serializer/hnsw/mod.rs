mod lazy_item;
mod lazy_item_array;
mod neighbors;
mod node;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::HNSWIndexCache,
    prob_lazy_load::lazy_item::FileIndex,
    versioning::Hash,
};

pub trait HNSWIndexSerialize: Sized {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<Self, BufIoError>;
}
