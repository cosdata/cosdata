pub mod cosine;
pub mod dotproduct;
pub mod euclidean;
pub mod hamming;

use crate::storage::Storage;

pub trait DistanceFunction: std::fmt::Debug + Send + Sync {
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<f32, DistanceError>;
}

#[derive(Debug)]
pub enum DistanceError {
    StorageMismatch,
    CalculationError,
}
