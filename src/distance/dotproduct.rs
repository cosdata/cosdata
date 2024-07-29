use super::{DistanceError, DistanceFunction};
use crate::storage::Storage;
use half::f16;

pub struct DotProductDistance;

impl DistanceFunction for DotProductDistance {
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<f32, DistanceError> {
        match (x, y) {
            (
                Storage::UnsignedByte {
                    quant_vec: vec_x, ..
                },
                Storage::UnsignedByte {
                    quant_vec: vec_y, ..
                },
            ) => Ok(dot_product_u8(vec_x, vec_y) as f32),
            (
                Storage::HalfPrecisionFP {
                    quant_vec: vec_x, ..
                },
                Storage::HalfPrecisionFP {
                    quant_vec: vec_y, ..
                },
            ) => Ok(dot_product_f16(vec_x, vec_y)),
            (Storage::SubByte { .. }, Storage::SubByte { .. }) => {
                Err(DistanceError::CalculationError) // Implement if needed
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}

pub fn dot_product_u8(x: &[u8], y: &[u8]) -> u32 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a as u32) * (b as u32))
        .sum()
}

pub fn dot_product_f16(x: &[f16], y: &[f16]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| f32::from(a) * f32::from(b))
        .sum()
}
