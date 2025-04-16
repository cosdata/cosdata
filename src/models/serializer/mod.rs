pub mod hnsw;
pub mod inverted;
pub mod tf_idf;

mod metric_distance;
mod page;
mod pagepool;
mod quotients_map;
mod sparse_embedding;
mod storage;
mod tf_idf_document;
mod tree_map;
mod versioned_item;
mod versioned_pagepool;
mod versioned_vec;

#[cfg(test)]
mod tests;

use super::buffered_io::{BufIoError, BufferManager, BufferManagerFactory};
use super::types::FileOffset;

pub trait SimpleSerialize: Sized {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError>;
    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError>;
}

pub trait PartitionedSerialize: Sized {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<u8>,
        file_parts: u8,
        file_idx: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError>;

    fn deserialize(
        bufmans: &BufferManagerFactory<u8>,
        file_parts: u8,
        file_idx: u8,
        file_offset: FileOffset,
    ) -> Result<Self, BufIoError>;
}

impl SimpleSerialize for u16 {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        Ok(bufman.write_to_end_of_file(cursor, &self.to_le_bytes())? as u32)
    }

    fn deserialize(
        bufman: &BufferManager,
        FileOffset(offset): FileOffset,
    ) -> Result<Self, BufIoError>
    where
        Self: Sized,
    {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset as u64)?;
        let res = bufman.read_u16_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(res)
    }
}
