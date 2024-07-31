use super::{DistanceError, DistanceFunction};
use crate::storage::Storage;

#[derive(Debug)]
pub struct CosineDistance;

impl DistanceFunction for CosineDistance {
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
                // Implement cosine similarity for UnsignedByte storage
                unimplemented!("Cosine similarity for UnsignedByte not implemented yet")
            }
            (Storage::SubByte { .. }, Storage::SubByte { .. }) => {
                // Implement cosine similarity for SubByte storage
                unimplemented!("Cosine similarity for SubByte not implemented yet")
            }
            (Storage::HalfPrecisionFP { .. }, Storage::HalfPrecisionFP { .. }) => {
                // Implement cosine similarity for HalfPrecisionFP storage
                unimplemented!("Cosine similarity for HalfPrecisionFP not implemented yet")
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}
