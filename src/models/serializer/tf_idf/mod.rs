mod atomic_array;
mod data;
mod lazy_item;
mod node;
mod term;
mod versioned_vec;

#[cfg(test)]
mod tests;

use std::sync::atomic::AtomicU32;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    types::FileOffset,
    versioning::VersionNumber,
};

pub const TF_IDF_INDEX_DATA_CHUNK_SIZE: usize = 4;

pub trait TFIDFIndexSerialize: Sized {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        offset_counter: &AtomicU32,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
        cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError>;
}
