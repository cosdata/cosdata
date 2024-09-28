use super::{Quantization, QuantizationError, StorageType};
use crate::models::common::quantize_to_u8_bits;
use crate::storage::Storage;
use half::f16;

#[derive(Debug)]
pub struct ScalarQuantization;

impl Quantization for ScalarQuantization {
    fn quantize(
        &self,
        vector: &[f32],
        storage_type: StorageType,
    ) -> Result<Storage, QuantizationError> {
        match storage_type {
            StorageType::UnsignedByte => {
                let (out_of_range, has_negative) =
                    vector.iter().fold((false, false), |(oor, neg), &x| {
                        (oor || x > 1.0 || x < -1.0, neg || x < 0.0)
                    });
                if out_of_range {
                    return Err(QuantizationError::InvalidInput(String::from(
                        "Values sent in vector for quantization are out of range [-1,+1]",
                    )));
                }
                let quant_vec: Vec<u8> = vector
                    .iter()
                    .map(|&x| {
                        let y = if has_negative { x + 1.0 } else { x };
                        (y * 255.0).round() as u8
                    })
                    .collect();
                let mag = quant_vec.iter().map(|&x| x as u32 * x as u32).sum();
                Ok(Storage::UnsignedByte { mag, quant_vec })
            }
            StorageType::SubByte(resolution) => {
                let quant_vec: Vec<_> = quantize_to_u8_bits(vector, resolution);
                let mag_sqr: f32 = vector.iter().map(|x| x * x).sum();
                let mag = mag_sqr.sqrt();
                Ok(Storage::SubByte {
                    mag,
                    quant_vec,
                    resolution,
                })
            }
            StorageType::HalfPrecisionFP => {
                let quant_vec = vector.iter().map(|&x| f16::from_f32(x)).collect();
                let mag = vector.iter().map(|&x| x * x).sum();
                Ok(Storage::HalfPrecisionFP { mag, quant_vec })
            }
        }
    }

    fn train(&mut self, _vectors: &[&[f32]]) -> Result<(), QuantizationError> {
        // Scalar quantization doesn't require training
        Ok(())
    }
}
