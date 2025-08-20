use std::{collections::HashMap, io};

use rustc_hash::FxHashMap;

use crate::{
    indexes::inverted::ZoneId,
    metadata::FieldValue,
    models::{
        buffered_io::{BufIoError, BufferManager},
        collection::{GeoFenceMetadata, RawVectorEmbedding},
        types::{DocumentId, FileOffset, VectorId},
    },
};

use super::{read_len, read_string, write_len, SimpleSerialize};

impl SimpleSerialize for RawVectorEmbedding {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        let mut buf = Vec::new();
        let mut flags = 0u8;
        if self.document_id.is_some() {
            flags |= 1 << 0;
        }

        if self.dense_values.is_some() {
            flags |= 1 << 1;
        }

        if self.metadata.is_some() {
            flags |= 1 << 2;
        }

        if self.geo_fence_values.is_some() {
            flags |= 1 << 3;
        }

        if self.geo_fence_metadata.is_some() {
            flags |= 1 << 4;
        }

        if self.text.is_some() {
            flags |= 1 << 5;
        }
        buf.push(flags);
        write_len(&mut buf, self.id.len() as u32);
        buf.extend(self.id.as_bytes());

        if let Some(document_id) = &self.document_id {
            write_len(&mut buf, document_id.len() as u32);
            buf.extend(document_id.as_bytes());
        }

        if let Some(dense_values) = &self.dense_values {
            write_len(&mut buf, dense_values.len() as u32);
            for val in dense_values {
                buf.extend(val.to_le_bytes());
            }
        }

        if let Some(metadata) = &self.metadata {
            write_len(&mut buf, metadata.len() as u32);
            for (field, val) in metadata {
                write_len(&mut buf, field.len() as u32);
                buf.extend(field.as_bytes());

                match val {
                    FieldValue::Int(int) => {
                        buf.push(0);
                        buf.extend(int.to_le_bytes());
                    }
                    FieldValue::String(str) => {
                        buf.push(1);
                        write_len(&mut buf, str.len() as u32);
                        buf.extend(str.as_bytes());
                    }
                }
            }
        }

        if let Some(geo_fence_values) = &self.geo_fence_values {
            write_len(&mut buf, geo_fence_values.len() as u32);
            for (field, value) in geo_fence_values {
                write_len(&mut buf, field.len() as u32);
                buf.extend(field.as_bytes());
                write_len(&mut buf, value.len() as u32);
                buf.extend(value.as_bytes());
            }
        }

        if let Some(geo_fence_metadata) = &self.geo_fence_metadata {
            buf.extend(geo_fence_metadata.weight.to_le_bytes());
            buf.extend(geo_fence_metadata.coordinates.0.to_le_bytes());
            buf.extend(geo_fence_metadata.coordinates.1.to_le_bytes());
            write_len(&mut buf, geo_fence_metadata.zone.len() as u32);
            buf.extend(geo_fence_metadata.zone.as_bytes());
        }

        if let Some(text) = &self.text {
            write_len(&mut buf, text.len() as u32);
            buf.extend(text.as_bytes());
        }

        Ok(bufman.write_to_end_of_file(cursor, &buf)? as u32)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;
        let flags = bufman.read_u8_with_cursor(cursor)?;
        let id = VectorId::from(read_string(bufman, cursor)?);

        let document_id = if (flags & 1) != 0 {
            Some(DocumentId::from(read_string(bufman, cursor)?))
        } else {
            None
        };

        let dense_values = if (flags & (1 << 1)) != 0 {
            let dense_values_len = read_len(bufman, cursor)? as usize;
            let mut values = Vec::with_capacity(dense_values_len);
            for _ in 0..dense_values_len {
                values.push(bufman.read_f32_with_cursor(cursor)?);
            }
            Some(values)
        } else {
            None
        };

        let metadata = if (flags & (1 << 2)) != 0 {
            let metadata_len = read_len(bufman, cursor)? as usize;
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
        } else {
            None
        };

        let geo_fence_values = if (flags & (1 << 3)) != 0 {
            let geo_fence_values_len = read_len(bufman, cursor)? as usize;
            let mut geo_fence_values = FxHashMap::default();

            for _ in 0..geo_fence_values_len {
                let field = read_string(bufman, cursor)?;
                let value = read_string(bufman, cursor)?;
                geo_fence_values.insert(field, value);
            }

            Some(geo_fence_values)
        } else {
            None
        };

        let geo_fence_metadata = if (flags & (1 << 4)) != 0 {
            let weight = bufman.read_f32_with_cursor(cursor)?;
            let coordinates = (
                bufman.read_f32_with_cursor(cursor)?,
                bufman.read_f32_with_cursor(cursor)?,
            );
            let zone = ZoneId::from(read_string(bufman, cursor)?);
            Some(GeoFenceMetadata {
                weight,
                coordinates,
                zone,
            })
        } else {
            None
        };

        let text = if (flags & (1 << 5)) != 0 {
            Some(read_string(bufman, cursor)?)
        } else {
            None
        };

        Ok(Self {
            id,
            document_id,
            dense_values,
            metadata,
            geo_fence_values,
            geo_fence_metadata,
            text,
        })
    }
}
