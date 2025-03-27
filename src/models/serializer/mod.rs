pub mod hnsw;
pub mod inverted;
pub mod inverted_idf;

mod metric_distance;
mod page;
mod pagepool;
mod storage;
mod versioned_pagepool;

#[cfg(test)]
mod tests;

use super::buffered_io::{BufIoError, BufferManager};
use super::types::FileOffset;

#[allow(unused)]
trait SimpleSerialize: Sized {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError>;
    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError>;
}

impl SimpleSerialize for f32 {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let offset = bufman.cursor_position(cursor)? as u32;
        bufman.update_f32_with_cursor(cursor, *self)?;
        Ok(offset)
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
        let res = bufman.read_f32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(res)
    }
}

impl SimpleSerialize for u32 {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let offset = bufman.cursor_position(cursor)? as u32;
        bufman.update_u32_with_cursor(cursor, *self)?;
        Ok(offset)
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
        let res = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(res)
    }
}
