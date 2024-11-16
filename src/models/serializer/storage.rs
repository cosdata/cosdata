use std::{
    io::{self, SeekFrom},
    sync::Arc,
};

use half::f16;

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManager},
        types::FileOffset,
    },
    storage::Storage,
};

use super::SimpleSerialize;

impl SimpleSerialize for Storage {
    fn serialize(&self, bufman: Arc<BufferManager>, cursor: u64) -> Result<u32, BufIoError> {
        let start = bufman.cursor_position(cursor)? as u32;

        match self {
            Self::UnsignedByte { mag, quant_vec } => {
                bufman.write_u8_with_cursor(cursor, 0)?;
                bufman.write_u32_with_cursor(cursor, *mag)?;
                bufman.write_u32_with_cursor(cursor, quant_vec.len() as u32)?;
                for el in quant_vec {
                    bufman.write_u8_with_cursor(cursor, *el)?;
                }
            }
            Self::SubByte {
                mag,
                quant_vec,
                resolution,
            } => {
                bufman.write_u8_with_cursor(cursor, 1)?;
                bufman.write_u8_with_cursor(cursor, *resolution)?;
                bufman.write_f32_with_cursor(cursor, *mag)?;
                bufman.write_u32_with_cursor(cursor, quant_vec.len() as u32)?;
                for vec in quant_vec {
                    bufman.write_u32_with_cursor(cursor, vec.len() as u32)?;
                    for el in vec {
                        bufman.write_u8_with_cursor(cursor, *el)?;
                    }
                }
            }
            Self::HalfPrecisionFP { mag, quant_vec } => {
                bufman.write_u8_with_cursor(cursor, 2)?;
                bufman.write_f32_with_cursor(cursor, *mag)?;
                bufman.write_u32_with_cursor(cursor, quant_vec.len() as u32)?;

                for el in quant_vec {
                    bufman.write_with_cursor(cursor, &el.to_le_bytes())?;
                }
            }
        }

        Ok(start)
    }

    fn deserialize(
        bufman: Arc<BufferManager>,
        FileOffset(offset): FileOffset,
    ) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, SeekFrom::Start(offset as u64))?;

        let variant_index = bufman.read_u8_with_cursor(cursor)?;

        let storage = match variant_index {
            0 => {
                let mag = bufman.read_u32_with_cursor(cursor)?;
                let len = bufman.read_u32_with_cursor(cursor)? as usize;
                let mut quant_vec = Vec::with_capacity(len);

                for _ in 0..len {
                    let el = bufman.read_u8_with_cursor(cursor)?;
                    quant_vec.push(el);
                }

                Self::UnsignedByte { mag, quant_vec }
            }
            1 => {
                let resolution = bufman.read_u8_with_cursor(cursor)?;
                let mag = bufman.read_f32_with_cursor(cursor)?;
                let len = bufman.read_u32_with_cursor(cursor)? as usize;
                let mut quant_vec = Vec::with_capacity(len);

                for _ in 0..len {
                    let len = bufman.read_u32_with_cursor(cursor)? as usize;
                    let mut vec = Vec::with_capacity(len);
                    for _ in 0..len {
                        let el = bufman.read_u8_with_cursor(cursor)?;
                        vec.push(el);
                    }
                    quant_vec.push(vec);
                }

                Self::SubByte {
                    mag,
                    quant_vec,
                    resolution,
                }
            }
            2 => {
                let mag = bufman.read_f32_with_cursor(cursor)?;
                let len = bufman.read_u32_with_cursor(cursor)? as usize;
                let mut quant_vec = Vec::with_capacity(len);

                for _ in 0..len {
                    let mut bytes = [0; 2];
                    bufman.read_with_cursor(cursor, &mut bytes)?;
                    let el = f16::from_le_bytes(bytes);
                    quant_vec.push(el);
                }

                Self::HalfPrecisionFP { mag, quant_vec }
            }
            _ => {
                return Err(
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid Storage variant").into(),
                );
            }
        };

        bufman.close_cursor(cursor)?;
        Ok(storage)
    }
}
