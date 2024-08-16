pub mod cosine;
pub mod dotproduct;
pub mod euclidean;
pub mod hamming;

use crate::storage::Storage;

pub trait DistanceFunction: std::fmt::Debug + Send + Sync {
    type Item;
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<Self::Item, DistanceError>;
}

#[derive(Debug)]
pub enum DistanceError {
    StorageMismatch,
    CalculationError,
}
