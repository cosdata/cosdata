mod lazy_item;
mod neighbors;
mod node;
#[cfg(test)]
mod tests;

use rustc_hash::FxHashMap;

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::{BufIoError, BufferManager},
        cache_loader::HNSWIndexCache,
        prob_lazy_load::lazy_item::FileIndex,
        prob_node::SharedNode,
        types::FileOffset,
    },
};

pub trait HNSWIndexSerialize: Sized {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError>;

    fn deserialize(
        bufman: &BufferManager,
        offset: FileOffset,
        file_id: IndexFileId,
        cache: &HNSWIndexCache,
        ready_items: &FxHashMap<FileIndex, SharedNode>,
        pending_items: &mut FxHashMap<FileIndex, SharedNode>,
    ) -> Result<Self, BufIoError>;
}
