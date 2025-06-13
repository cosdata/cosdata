use half::f16;
use serde::{Deserialize, Serialize};

use super::{DistanceError, DistanceFunction};
use crate::{models::types::VectorData, storage::Storage};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize, PartialOrd)]
pub struct HammingDistance(pub f32);

impl DistanceFunction for HammingDistance {
    type Item = Self;

    // Implementation here
    #[allow(unused_variables)]
    fn calculate(
        &self,
        x: &VectorData,
        y: &VectorData,
        _is_indexing: bool,
    ) -> Result<Self::Item, DistanceError> {
        match (x.quantized_vec, y.quantized_vec) {
            (
                Storage::UnsignedByte {
                    quant_vec: vec_x, ..
                },
                Storage::UnsignedByte {
                    quant_vec: vec_y, ..
                },
            ) => Ok(hamming_distance_u8(vec_x, vec_y)),
            (
                Storage::SubByte {
                    quant_vec: vec_x,
                    resolution: res_x,
                    ..
                },
                Storage::SubByte {
                    quant_vec: vec_y,
                    resolution: res_y,
                    ..
                },
            ) => {
                if res_x != res_y {
                    return Err(DistanceError::StorageMismatch);
                }
                Ok(hamming_distance_subbyte(vec_x, vec_y, *res_x))
            }
            (
                Storage::HalfPrecisionFP {
                    quant_vec: vec_x, ..
                },
                Storage::HalfPrecisionFP {
                    quant_vec: vec_y, ..
                },
            ) => Ok(hamming_distance_f16(vec_x, vec_y)),
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}

pub fn hamming_distance_u8(x: &[u8], y: &[u8]) -> HammingDistance {
    if x.len() != y.len() {
        return HammingDistance(f32::INFINITY);
    }

    let distance = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a ^ b).count_ones() as f32)
        .sum::<f32>();
    HammingDistance(distance)
}

pub fn hamming_distance_subbyte(x: &[Vec<u8>], y: &[Vec<u8>], resolution: u8) -> HammingDistance {
    if x.len() != y.len() || resolution == 0 || resolution > 8 {
        return HammingDistance(f32::INFINITY);
    }

    let mask = (1u8 << resolution) - 1;
    let mut total_distance = 0f32;

    for (vec_x, vec_y) in x.iter().zip(y.iter()) {
        if vec_x.len() != vec_y.len() {
            return HammingDistance(f32::INFINITY);
        }

        for (&byte_x, &byte_y) in vec_x.iter().zip(vec_y.iter()) {
            let bits_per_byte = 8 / resolution;
            for i in 0..bits_per_byte {
                let shift = i * resolution;
                let val_x = (byte_x >> shift) & mask;
                let val_y = (byte_y >> shift) & mask;
                total_distance += (val_x ^ val_y).count_ones() as f32;
            }
        }
    }

    HammingDistance(total_distance)
}

pub fn hamming_distance_f16(x: &[f16], y: &[f16]) -> HammingDistance {
    if x.len() != y.len() {
        return HammingDistance(f32::INFINITY);
    }

    let distance = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| {
            let bits_a = a.to_bits();
            let bits_b = b.to_bits();
            (bits_a ^ bits_b).count_ones() as f32
        })
        .sum::<f32>();
    HammingDistance(distance)
}
