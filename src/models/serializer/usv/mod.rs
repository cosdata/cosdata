mod atomic_array;
mod data;
mod id_lists;
mod lazy_item;
mod node;
mod versioned_vec;

#[cfg(test)]
mod tests;

use std::sync::atomic::AtomicU32;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::USVIndexCache,
    types::FileOffset,
    versioning::VersionNumber,
};

pub const USV_INDEX_DATA_CHUNK_SIZE: usize = 4;

pub trait USVIndexSerialize: Sized {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        quantization_bits: u8,
        offset_counter: &AtomicU32,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
        cache: &USVIndexCache,
    ) -> Result<Self, BufIoError>;
}
