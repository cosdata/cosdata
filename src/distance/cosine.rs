use serde::{Deserialize, Serialize};

use super::{DistanceError, DistanceFunction};
use crate::{
    models::dot_product::{
        dot_product_binary, dot_product_f16, dot_product_octal, dot_product_quaternary,
        dot_product_u8,
    },
    storage::Storage,
};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub struct CosineDistance(pub f32);

impl DistanceFunction for CosineDistance {
    type Item = Self;
    fn calculate(&self, _x: &Storage, _y: &Storage) -> Result<Self::Item, DistanceError> {
        // TODO: Implement cosine distance
        unimplemented!("Cosine distance is not implemented yet");
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub struct CosineSimilarity(pub f32);

impl DistanceFunction for CosineSimilarity {
    type Item = Self;
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<Self::Item, DistanceError> {
        match (x, y) {
            (
                Storage::UnsignedByte {
                    mag: x_mag,
                    quant_vec: x_vec,
                },
                Storage::UnsignedByte {
                    mag: y_mag,
                    quant_vec: y_vec,
                },
            ) => {
                let dot_product = dot_product_u8(x_vec, y_vec) as f32;
                let x_mag = (*x_mag as f32).sqrt();
                let y_mag = (*y_mag as f32).sqrt();
                cosine_similarity_from_dot_product(dot_product, x_mag, y_mag)
            }
            (
                Storage::SubByte {
                    mag: x_mag,
                    quant_vec: x_vec,
                    resolution: x_res,
                },
                Storage::SubByte {
                    mag: y_mag,
                    quant_vec: y_vec,
                    resolution: y_res,
                },
            ) => {
                if x_res != y_res {
                    return Err(DistanceError::StorageMismatch);
                }
                let dot_product = match *x_res {
                    1 => dot_product_binary(x_vec, y_vec, *x_res),
                    2 => dot_product_quaternary(x_vec, y_vec, *x_res),
                    3 => dot_product_octal(x_vec, y_vec, *x_res),
                    _ => {
                        return Err(DistanceError::CalculationError);
                    }
                };
                cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag)
            }
            (
                Storage::HalfPrecisionFP {
                    mag: x_mag,
                    quant_vec: x_vec,
                },
                Storage::HalfPrecisionFP {
                    mag: y_mag,
                    quant_vec: y_vec,
                },
            ) => {
                let dot_product = dot_product_f16(x_vec, y_vec);
                cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag)
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}

fn cosine_similarity_from_dot_product(
    dot_product: f32,
    x_mag: f32,
    y_mag: f32,
) -> Result<CosineSimilarity, DistanceError> {
    let denominator = x_mag * y_mag;

    if denominator == 0.0 {
        Err(DistanceError::CalculationError)
    } else {
        Ok(CosineSimilarity(dot_product / denominator))
    }
}
