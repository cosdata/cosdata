pub mod hnsw;
pub mod inverted;
pub mod tf_idf;

mod metric_distance;
mod quotients_map;
mod raw_vector_embedding;
mod storage;
mod transaction_status;
mod tree_map;
mod versioned_item;
mod versioned_vec;

#[cfg(test)]
mod tests;

use std::io;

use parking_lot::RwLock;

use super::buffered_io::{BufIoError, BufferManager, BufferManagerFactory};
use super::types::{DocumentId, FileOffset, InternalId, VectorId};

/// Writes a u16 length to the buffer using custom encoding:
/// - If the value fits in 7 bits (<= 127), write it directly as one byte.
/// - Otherwise, set the MSB of the first byte to 1, store the top 7 bits,
///   and follow with a byte for the lower 8 bits.
pub fn write_len(buf: &mut Vec<u8>, len: u16) {
    if len <= 0x7F {
        buf.push(len as u8);
    } else {
        let msb = ((len >> 8) as u8 & 0x7F) | 0x80; // Set highest bit
        let lsb = len as u8;
        buf.push(msb);
        buf.push(lsb);
    }
}

/// Reads a u16 length from the buffer using the same encoding.
pub fn read_len(bufman: &BufferManager, cursor: u64) -> Result<u16, BufIoError> {
    let first = bufman.read_u8_with_cursor(cursor)?;
    if first & 0x80 == 0 {
        Ok(first as u16)
    } else {
        let msb = (first & 0x7F) as u16;
        let lsb = bufman.read_u8_with_cursor(cursor)? as u16;
        Ok((msb << 8) | lsb)
    }
}

pub fn read_string(bufman: &BufferManager, cursor: u64) -> Result<String, BufIoError> {
    let len = read_len(bufman, cursor)? as usize;
    let mut buf = vec![0; len];
    bufman.read_with_cursor(cursor, &mut buf)?;
    String::from_utf8(buf)
        .map_err(|err| BufIoError::Io(io::Error::new(io::ErrorKind::InvalidData, err)))
}

pub fn read_opt_string(bufman: &BufferManager, cursor: u64) -> Result<Option<String>, BufIoError> {
    let len = read_len(bufman, cursor)? as usize;
    if len == 0 {
        return Ok(None);
    }
    let mut buf = vec![0; len];
    bufman.read_with_cursor(cursor, &mut buf)?;
    let str = String::from_utf8(buf)
        .map_err(|err| BufIoError::Io(io::Error::new(io::ErrorKind::InvalidData, err)))?;
    Ok(Some(str))
}

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

impl SimpleSerialize for InternalId {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        Ok(bufman.write_to_end_of_file(cursor, &self.to_le_bytes())? as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let id = Self::from(bufman.read_u32_with_cursor(cursor)?);
        bufman.close_cursor(cursor)?;
        Ok(id)
    }
}

impl SimpleSerialize for VectorId {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut buf = Vec::new();
        write_len(&mut buf, self.len() as u16);
        buf.extend(self.as_bytes());
        Ok(bufman.write_to_end_of_file(cursor, &buf)? as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let id = read_string(bufman, cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(Self::from(id))
    }
}

impl SimpleSerialize for DocumentId {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut buf = Vec::new();
        write_len(&mut buf, self.len() as u16);
        buf.extend(self.as_bytes());
        Ok(bufman.write_to_end_of_file(cursor, &buf)? as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let id = read_string(bufman, cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(Self::from(id))
    }
}

impl SimpleSerialize for u64 {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        Ok(bufman.write_to_end_of_file(cursor, &self.to_le_bytes())? as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let val = bufman.read_u64_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(val)
    }
}

impl SimpleSerialize for (u32, f32) {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut buf = Vec::with_capacity(8);
        buf.extend(self.0.to_le_bytes());
        buf.extend(self.1.to_le_bytes());
        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        Ok(offset)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let idx = bufman.read_u32_with_cursor(cursor)?;
        let val = bufman.read_f32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok((idx, val))
    }
}

impl<T: SimpleSerialize> SimpleSerialize for RwLock<T> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        self.read().serialize(bufman, cursor)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        Ok(Self::new(T::deserialize(bufman, offset)?))
    }
}

impl SimpleSerialize for u32 {
    fn serialize(&self, _bufman: &BufferManager, _cursor: u64) -> Result<u32, BufIoError> {
        Ok(*self)
    }

    fn deserialize(_bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        Ok(offset.0)
    }
}
