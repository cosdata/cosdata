use super::{DistanceError, DistanceFunction};
use crate::models::dot_product::dot_product_u8;
use crate::models::types::MetricResult;
use crate::storage::Storage;
use half::f16;

#[derive(Debug)]
pub struct DotProductDistance;

impl DistanceFunction for DotProductDistance {
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<MetricResult, DistanceError> {
        match (x, y) {
            (
                Storage::UnsignedByte {
                    quant_vec: vec_x, ..
                },
                Storage::UnsignedByte {
                    quant_vec: vec_y, ..
                },
            ) => Ok(MetricResult::DotProductDistance(
                dot_product_u8(vec_x, vec_y) as f32,
            )),
            (
                Storage::HalfPrecisionFP {
                    quant_vec: vec_x, ..
                },
                Storage::HalfPrecisionFP {
                    quant_vec: vec_y, ..
                },
            ) => Ok(MetricResult::DotProductDistance(dot_product_f16(
                vec_x, vec_y,
            ))),
            (Storage::SubByte { .. }, Storage::SubByte { .. }) => {
                Err(DistanceError::CalculationError) // Implement if needed
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}

pub fn dot_product_f16(x: &[f16], y: &[f16]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| f32::from(a) * f32::from(b))
        .sum()
}
