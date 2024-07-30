use super::{DistanceError, DistanceFunction};
use crate::storage::Storage;

#[derive(Debug)]
pub struct HammingDistance;

impl DistanceFunction for HammingDistance {
    // Implementation here
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<f32, DistanceError> {
        todo!()
    }
}
