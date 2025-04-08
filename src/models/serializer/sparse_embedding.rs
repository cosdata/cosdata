use std::sync::Arc;

use crate::{
    indexes::inverted::types::{RawSparseVectorEmbedding, SparsePair},
    models::{
        buffered_io::{BufIoError, BufferManager},
        types::{FileOffset, VectorId},
    },
};

use super::SimpleSerialize;

impl SimpleSerialize for RawSparseVectorEmbedding {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let big_len = self.raw_vec.len() > 127;
        let mut buf = Vec::with_capacity(8 + if big_len { 2 } else { 1 } + 8 * self.raw_vec.len());
        buf.extend(self.hash_vec.0.to_le_bytes());
        buf.extend((self.raw_vec.len() as u32).to_le_bytes());
        if big_len {
            let len = (self.raw_vec.len() as u16) & (1 << 15);
            buf.extend(len.to_be_bytes());
        } else {
            buf.push(self.raw_vec.len() as u8);
        }

        for pair in self.raw_vec.iter() {
            buf.extend(pair.0.to_le_bytes());
            buf.extend(pair.1.to_le_bytes());
        }

        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;

        Ok(offset)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        bufman.close_cursor(cursor)?;
        let hash_vec = VectorId(bufman.read_u64_with_cursor(cursor)?);
        let len = {
            let first_byte = bufman.read_u8_with_cursor(cursor)?;
            if first_byte & (1 << 7) == 0 {
                first_byte as usize
            } else {
                let second_byte = bufman.read_u8_with_cursor(cursor)?;
                u16::from_be_bytes([(first_byte & 0x7F), second_byte]) as usize
            }
        };
        let mut raw_vec = Vec::with_capacity(len);

        for _ in 0..len {
            let dim = bufman.read_u32_with_cursor(cursor)?;
            let value = bufman.read_f32_with_cursor(cursor)?;
            let pair = SparsePair(dim, value);
            raw_vec.push(pair);
        }

        Ok(Self {
            raw_vec: Arc::new(raw_vec),
            hash_vec,
        })
    }
}
