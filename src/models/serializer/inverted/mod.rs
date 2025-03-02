mod atomic_array;
mod data;
mod lazy_item;
mod node;
mod page;
mod pagepool;
mod versioned_fixedsets;
mod versioned_pagepool;

#[cfg(test)]
mod tests;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    types::FileOffset,
};

pub const DATA_FILE_PARTS: u32 = 8;

pub trait InvertedIndexSerialize: Sized {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError>;
}
