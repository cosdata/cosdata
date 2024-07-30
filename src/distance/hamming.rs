use super::{DistanceError, DistanceFunction};
use crate::storage::Storage;

#[derive(Debug)]
pub struct HammingDistance;

impl DistanceFunction for HammingDistance {
    // Implementation here
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<f32, DistanceError> {
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
                // Implement hamming similarity for UnsignedByte storage
                unimplemented!("Hamming similarity for UnsignedByte not implemented yet")
            }
            (Storage::SubByte { .. }, Storage::SubByte { .. }) => {
                // Implement hamming similarity for SubByte storage
                unimplemented!("Hamming similarity for SubByte not implemented yet")
            }
            (Storage::HalfPrecisionFP { .. }, Storage::HalfPrecisionFP { .. }) => {
                // Implement hamming similarity for HalfPrecisionFP storage
                unimplemented!("Hamming similarity for HalfPrecisionFP not implemented yet")
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}
