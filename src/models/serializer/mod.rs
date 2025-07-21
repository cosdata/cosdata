pub mod hnsw;
pub mod inverted;
pub mod tf_idf;
pub mod tree_map;

mod metric_distance;
mod raw_vector_embedding;
mod storage;
mod transaction_status;

#[cfg(test)]
mod tests;

use std::io;

use parking_lot::RwLock;

use super::buffered_io::{BufIoError, BufferManager};
use super::types::{DocumentId, FileOffset, InternalId, VectorId};

/// Encode `len` into 1–3 bytes:
/// - 1 byte if < 2⁷  
/// - 2 bytes if < 2¹⁴  
/// - 3 bytes otherwise (uses all 8 bits of the final byte, giving 22 bits total)
pub fn write_len(buf: &mut Vec<u8>, len: u32) {
    if len < (1 << 7) {
        // [0vvvvvvv]
        buf.push(len as u8);
    } else if len < (1 << 14) {
        // [1vvvvvvv] [0vvvvvvv]
        buf.push(((len) as u8 & 0x7F) | 0x80);
        buf.push((len >> 7) as u8);
    } else {
        // [1vvvvvvv] [1vvvvvvv] [vvvvvvvv]
        buf.push(((len) as u8 & 0x7F) | 0x80);
        buf.push(((len >> 7) as u8 & 0x7F) | 0x80);
        buf.push((len >> 14) as u8);
    }
}

/// Decode a 1–3 byte varint back to `u32`. Any bits beyond 22 are ignored.
pub fn read_len(bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
    let b0 = bufman.read_u8_with_cursor(cursor)? as u32;
    if b0 & 0x80 == 0 {
        return Ok(b0);
    }

    let b1 = bufman.read_u8_with_cursor(cursor)? as u32;
    let low14 = (b0 & 0x7F) | ((b1 & 0x7F) << 7);
    if b1 & 0x80 == 0 {
        return Ok(low14);
    }

    let b2 = bufman.read_u8_with_cursor(cursor)? as u32;
    // here we take all 8 bits of b2 as the highest part
    Ok(low14 | (b2 << 14))
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
        write_len(&mut buf, self.len() as u32);
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
        write_len(&mut buf, self.len() as u32);
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
