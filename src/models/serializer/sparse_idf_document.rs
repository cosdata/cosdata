use std::io;

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    types::{FileOffset, VectorId},
};

use super::SimpleSerialize;

impl SimpleSerialize for (VectorId, Option<String>) {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let text_serialized_size = if let Some(text) = &self.1 {
            let len_size = if text.len() > 127 { 2 } else { 1 };
            len_size + text.len()
        } else {
            2
        };
        let mut buf = Vec::with_capacity(8 + text_serialized_size);
        buf.extend(self.0 .0.to_be_bytes());
        if let Some(text) = &self.1 {
            if text.len() > 127 {
                let len = (text.len() as u16) & (1 << 15);
                buf.extend(len.to_be_bytes());
            } else {
                buf.push(text.len() as u8);
            }
            buf.extend(text.as_bytes());
        } else {
            buf.extend([u8::MAX; 2]);
        }
        let offset = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        Ok(offset)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let id = VectorId(bufman.read_u64_with_cursor(cursor)?);
        let len = {
            let first_byte = bufman.read_u8_with_cursor(cursor)?;
            if first_byte & (1 << 7) == 0 {
                first_byte as usize
            } else {
                let second_byte = bufman.read_u8_with_cursor(cursor)?;
                u16::from_be_bytes([(first_byte & 0x7F), second_byte]) as usize
            }
        };
        let text = if len == u16::MAX as usize {
            None
        } else {
            let mut buf = vec![0u8; len];
            bufman.read_with_cursor(cursor, &mut buf)?;
            let text = String::from_utf8(buf)
                .map_err(|err| BufIoError::Io(io::Error::new(io::ErrorKind::InvalidData, err)))?;
            Some(text)
        };
        bufman.close_cursor(cursor)?;

        Ok((id, text))
    }
}
