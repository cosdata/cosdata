use super::{DistanceError, DistanceFunction};
use crate::storage::Storage;
use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub struct EuclideanDistance(pub f32);

impl DistanceFunction for EuclideanDistance {
    type Item = Self;
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<Self::Item, DistanceError> {
        match (x, y) {
            (
                Storage::UnsignedByte {
                    quant_vec: vec_x, ..
                },
                Storage::UnsignedByte {
                    quant_vec: vec_y, ..
                },
            ) => Ok(euclidean_distance_u8(vec_x, vec_y)),
            (
                Storage::HalfPrecisionFP {
                    quant_vec: vec_x, ..
                },
                Storage::HalfPrecisionFP {
                    quant_vec: vec_y, ..
                },
            ) => Ok(euclidean_distance_f16(vec_x, vec_y)),
            (Storage::SubByte { .. }, Storage::SubByte { .. }) => {
                // TODO: Implement euclidean distance for SubByte storage
                unimplemented!("Euclidean distance for SubByte is not implemented yet");
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}
pub fn euclidean_distance_u8(x: &[u8], y: &[u8]) -> EuclideanDistance {
    EuclideanDistance(
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| {
                let diff = (a as i16) - (b as i16);
                (diff * diff) as f32
            })
            .sum::<f32>()
            .sqrt(),
    )
}

pub fn euclidean_distance_f16(x: &[f16], y: &[f16]) -> EuclideanDistance {
    EuclideanDistance(
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| {
                let diff = f32::from(a) - f32::from(b);
                diff * diff
            })
            .sum::<f32>()
            .sqrt(),
    )
}
