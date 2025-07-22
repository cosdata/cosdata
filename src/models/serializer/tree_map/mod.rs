mod node;
mod quotients_map;
mod versioned_item;
mod versioned_vec;

#[cfg(test)]
mod tests;

use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    types::FileOffset,
    versioning::VersionNumber,
};

pub trait TreeMapSerialize: Sized {
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
    ) -> Result<Self, BufIoError>;
}
