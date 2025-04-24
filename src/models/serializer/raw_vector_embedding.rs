use std::{collections::HashMap, io};

use crate::{
    indexes::inverted::types::SparsePair,
    metadata::FieldValue,
    models::{
        buffered_io::{BufIoError, BufferManager},
        collection::RawVectorEmbedding,
        types::{DocumentId, FileOffset, VectorId},
    },
};

use super::SimpleSerialize;

/// Writes a u16 length to the buffer using custom encoding:
/// - If the value fits in 7 bits (<= 127), write it directly as one byte.
/// - Otherwise, set the MSB of the first byte to 1, store the top 7 bits,
///   and follow with a byte for the lower 8 bits.
fn write_len(buf: &mut Vec<u8>, len: u16) {
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
fn read_len(bufman: &BufferManager, cursor: u64) -> Result<u16, BufIoError> {
    let first = bufman.read_u8_with_cursor(cursor)?;
    if first & 0x80 == 0 {
        Ok(first as u16)
    } else {
        let msb = (first & 0x7F) as u16;
        let lsb = bufman.read_u8_with_cursor(cursor)? as u16;
        Ok((msb << 8) | lsb)
    }
}

fn read_string(bufman: &BufferManager, cursor: u64) -> Result<String, BufIoError> {
    let len = read_len(bufman, cursor)? as usize;
    let mut buf = vec![0; len];
    bufman.read_with_cursor(cursor, &mut buf)?;
    String::from_utf8(buf)
        .map_err(|err| BufIoError::Io(io::Error::new(io::ErrorKind::InvalidData, err)))
}

fn read_opt_string(bufman: &BufferManager, cursor: u64) -> Result<Option<String>, BufIoError> {
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

impl SimpleSerialize for RawVectorEmbedding {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut buf = Vec::new();
        write_len(&mut buf, self.id.len() as u16);
        buf.extend(self.id.as_bytes());
        if let Some(document_id) = &self.document_id {
            write_len(&mut buf, document_id.len() as u16);
            buf.extend(document_id.as_bytes());
        } else {
            write_len(&mut buf, 0);
        }
        if let Some(dense_values) = &self.dense_values {
            write_len(&mut buf, dense_values.len() as u16);
            for val in dense_values {
                buf.extend(val.to_le_bytes());
            }
        } else {
            write_len(&mut buf, 0);
        }
        if let Some(metadata) = &self.metadata {
            write_len(&mut buf, metadata.len() as u16);
            for (field, val) in metadata {
                write_len(&mut buf, field.len() as u16);
                buf.extend(field.as_bytes());

                match val {
                    FieldValue::Int(int) => {
                        buf.push(0);
                        buf.extend(int.to_le_bytes());
                    }
                    FieldValue::String(str) => {
                        buf.push(1);
                        write_len(&mut buf, str.len() as u16);
                        buf.extend(str.as_bytes());
                    }
                }
            }
        } else {
            write_len(&mut buf, 0);
        }

        if let Some(sparse_values) = &self.sparse_values {
            write_len(&mut buf, sparse_values.len() as u16);
            for pair in sparse_values {
                buf.extend(pair.0.to_le_bytes());
                buf.extend(pair.1.to_le_bytes());
            }
        } else {
            write_len(&mut buf, 0);
        }

        if let Some(text) = &self.text {
            write_len(&mut buf, text.len() as u16);
            buf.extend(text.as_bytes());
        } else {
            write_len(&mut buf, 0);
        }

        Ok(bufman.write_to_end_of_file(cursor, &buf)? as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let id = VectorId::from(read_string(bufman, cursor)?);
        let document_id = read_opt_string(bufman, cursor)?.map(DocumentId::from);
        let dense_values_len = read_len(bufman, cursor)? as usize;
        let dense_values = if dense_values_len == 0 {
            None
        } else {
            let mut values = Vec::with_capacity(dense_values_len);
            for _ in 0..dense_values_len {
                values.push(bufman.read_f32_with_cursor(cursor)?);
            }
            Some(values)
        };
        let metadata_len = read_len(bufman, cursor)? as usize;
        let metadata = if metadata_len == 0 {
            None
        } else {
            let mut metadata = HashMap::with_capacity(metadata_len);

            for _ in 0..metadata_len {
                let field = read_string(bufman, cursor)?;
                let val = {
                    let variant = bufman.read_u8_with_cursor(cursor)?;
                    match variant {
                        0 => FieldValue::Int(bufman.read_i32_with_cursor(cursor)?),
                        1 => FieldValue::String(read_string(bufman, cursor)?),
                        other => {
                            return Err(BufIoError::Io(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("Invalid FieldValue variant `{}`", other),
                            )))
                        }
                    }
                };
                metadata.insert(field, val);
            }

            Some(metadata)
        };

        let sparse_values_len = read_len(bufman, cursor)? as usize;
        let sparse_values = if sparse_values_len == 0 {
            None
        } else {
            let mut sparse_values = Vec::with_capacity(sparse_values_len);

            for _ in 0..sparse_values_len {
                let index = bufman.read_u32_with_cursor(cursor)?;
                let value = bufman.read_f32_with_cursor(cursor)?;
                let pair = SparsePair(index, value);
                sparse_values.push(pair);
            }

            Some(sparse_values)
        };

        let text = read_opt_string(bufman, cursor)?;

        Ok(Self {
            id,
            document_id,
            dense_values,
            metadata,
            sparse_values,
            text,
        })
    }
}
