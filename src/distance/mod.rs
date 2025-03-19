pub mod cosine;
pub mod dotproduct;
pub mod euclidean;
pub mod hamming;

use crate::models::types::VectorData;

pub trait DistanceFunction: std::fmt::Debug + Send + Sync {
    type Item;
    fn calculate(&self, x: &VectorData, y: &VectorData) -> Result<Self::Item, DistanceError>;
}

#[derive(Debug)]
pub enum DistanceError {
    StorageMismatch,
    CalculationError,
}
