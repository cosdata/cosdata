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
        range: (f32, f32),
    ) -> Result<Storage, QuantizationError> {
        match storage_type {
            StorageType::UnsignedByte => {
                let quant_vec: Vec<u8> = vector
                    .iter()
                    .map(|&x| {
                        (((x.max(range.0).min(range.1) - range.0) / (range.1 - range.0)) * 255.0)
                            as u8
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
