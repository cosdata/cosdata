mod lazy_item;
mod neighbors;
mod node;
#[cfg(test)]
mod tests;

use rustc_hash::FxHashSet;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager},
        cache_loader::HNSWIndexCache,
        prob_lazy_load::lazy_item::FileIndex,
        types::FileOffset,
    },
};

pub trait HNSWIndexSerialize: Sized {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError>;

    fn deserialize(
        bufman: &BufferManager,
        file_index: FileIndex,
        cache: &HNSWIndexCache,
        max_loads: u16,
        skipm: &mut FxHashSet<u64>,
    ) -> Result<Self, BufIoError>;
}

pub trait RawDeserialize: Sized {
    type Raw;

    fn deserialize_raw(
        bufman: &BufferManager,
        cursor: u64,
        offset: FileOffset,
        file_id: IndexFileId,
        cache: &HNSWIndexCache,
    ) -> Result<Self::Raw, BufIoError>;
}
