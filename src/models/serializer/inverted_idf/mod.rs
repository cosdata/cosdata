mod atomic_array;
mod data;
mod lazy_item;
mod node;
mod term;

#[cfg(test)]
mod tests;

use std::sync::atomic::AtomicU32;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexIDFCache,
    types::FileOffset,
};

pub const INVERTED_INDEX_DATA_CHUNK_SIZE: usize = 4;

pub trait InvertedIndexIDFSerialize: Sized {
    #[allow(clippy::too_many_arguments)]
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        quantization_bits: u8,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        quantization_bits: u8,
        data_file_idx: u8,
        data_file_parts: u8,
        cache: &InvertedIndexIDFCache,
    ) -> Result<Self, BufIoError>;
}
