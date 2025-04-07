mod atomic_array;
mod data;
mod lazy_item;
mod node;

#[cfg(test)]
mod tests;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    types::FileOffset,
};

pub trait InvertedIndexSerialize: Sized {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        data_file_parts: u8,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError>;
}
