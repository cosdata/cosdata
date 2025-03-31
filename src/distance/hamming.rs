use serde::{Deserialize, Serialize};

use super::{DistanceError, DistanceFunction};
use crate::{models::types::VectorData, storage::Storage};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize, PartialOrd)]
pub struct HammingDistance(pub f32);

impl DistanceFunction for HammingDistance {
    type Item = Self;

    // Implementation here
    #[allow(unused_variables)]
    fn calculate(&self, x: &VectorData, y: &VectorData, _is_indexing: bool) -> Result<Self::Item, DistanceError> {
        match (x.quantized_vec, y.quantized_vec) {
            (
                Storage::UnsignedByte {
                    mag: mag_x,
                    quant_vec: vec_x,
                },
                Storage::UnsignedByte {
                    mag: mag_y,
                    quant_vec: vec_y,
                },
            ) => {
                // TODO: Implement hamming similarity for UnsignedByte storage
                unimplemented!("Hamming similarity for UnsignedByte is not implemented yet");
            }
            (Storage::SubByte { .. }, Storage::SubByte { .. }) => {
                // TODO: Implement hamming similarity for SubByte storage
                unimplemented!("Hamming similarity for SubByte is not implemented yet");
            }
            (Storage::HalfPrecisionFP { .. }, Storage::HalfPrecisionFP { .. }) => {
                // TODO: Implement hamming similarity for HalfPrecisionFP storage
                unimplemented!("Hamming similarity for HalfPrecisionFP is not implemented yet");
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}
