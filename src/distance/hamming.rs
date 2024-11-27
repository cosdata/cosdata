use serde::{Deserialize, Serialize};

use super::{DistanceError, DistanceFunction};
use crate::storage::Storage;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub struct HammingDistance(pub f32);

impl DistanceFunction for HammingDistance {
    type Item = Self;

    // Implementation here
    #[allow(unused_variables)]
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<Self::Item, DistanceError> {
        match (x, y) {
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
