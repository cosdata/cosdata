use super::CustomSerialize;
use crate::models::types::VectorQt;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};

impl CustomSerialize for VectorQt {
    fn serialize<W: Write + Seek>(&mut self, writer: &mut W) -> std::io::Result<u32> {
        let offset = writer.stream_position()? as u32;

        match self {
            VectorQt::UnsignedByte { mag, quant_vec } => {
                writer.write_u8(0)?; // 0 for UnsignedByte
                writer.write_u32::<LittleEndian>(*mag)?;
                writer.write_u32::<LittleEndian>(quant_vec.len() as u32)?;
                writer.write_all(quant_vec)?;
            }
            VectorQt::SubByte {
                mag,
                quant_vec,
                resolution,
            } => {
                writer.write_u8(1)?; // 1 for SubByte
                writer.write_u32::<LittleEndian>(*mag)?;
                writer.write_u32::<LittleEndian>(quant_vec.len() as u32)?;
                for inner_vec in quant_vec {
                    writer.write_u32::<LittleEndian>(inner_vec.len() as u32)?;
                    for value in inner_vec {
                        writer.write_u32::<LittleEndian>(*value)?;
                    }
                }
                writer.write_u8(*resolution)?;
            }
        }

        Ok(offset)
    }

    fn deserialize<R: Read + Seek>(reader: &mut R, offset: u32) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(offset as u64))?;

        match reader.read_u8()? {
            0 => {
                let mag = reader.read_u32::<LittleEndian>()?;
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let mut quant_vec = vec![0u8; len];
                reader.read_exact(&mut quant_vec)?;
                Ok(VectorQt::UnsignedByte { mag, quant_vec })
            }
            1 => {
                let mag = reader.read_u32::<LittleEndian>()?;
                let outer_len = reader.read_u32::<LittleEndian>()? as usize;
                let mut quant_vec = Vec::with_capacity(outer_len);
                for _ in 0..outer_len {
                    let inner_len = reader.read_u32::<LittleEndian>()? as usize;
                    let mut inner_vec = Vec::with_capacity(inner_len);
                    for _ in 0..inner_len {
                        inner_vec.push(reader.read_u32::<LittleEndian>()?);
                    }
                    quant_vec.push(inner_vec);
                }
                let resolution = reader.read_u8()?;
                Ok(VectorQt::SubByte {
                    mag,
                    quant_vec,
                    resolution,
                })
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Null VectorQt",
            )),
        }
    }
}
