mod atomic_array;
mod data;
mod lazy_item;
mod node;
mod versioned_vec;

#[cfg(test)]
mod tests;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexCache,
    types::FileOffset,
    versioning::VersionNumber,
};

pub trait InvertedIndexSerialize: Sized {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<VersionNumber>,
        file_offset: FileOffset,
        version: VersionNumber,
        cache: &InvertedIndexCache,
    ) -> Result<Self, BufIoError>;
}
