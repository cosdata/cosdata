use super::{DistanceError, DistanceFunction};
use crate::models::dot_product::{
    dot_product_binary, dot_product_f16, dot_product_octal, dot_product_quaternary, dot_product_u8,
};
use crate::storage::Storage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub struct DotProductDistance(pub f32);

impl DistanceFunction for DotProductDistance {
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
            ) => Ok(DotProductDistance(dot_product_u8(vec_x, vec_y) as f32)),
            (
                Storage::HalfPrecisionFP {
                    quant_vec: vec_x, ..
                },
                Storage::HalfPrecisionFP {
                    quant_vec: vec_y, ..
                },
            ) => Ok(DotProductDistance(dot_product_f16(vec_x, vec_y))),
            (
                Storage::SubByte {
                    quant_vec: x_vec,
                    resolution: x_res,
                    ..
                },
                Storage::SubByte {
                    quant_vec: y_vec,
                    resolution: y_res,
                    ..
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
                Ok(DotProductDistance(dot_product))
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}
