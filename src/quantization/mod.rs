pub mod product;
pub mod scalar;

use crate::storage::Storage;

pub trait Quantization {
    fn quantize(&self, vector: &[f32], storage_type: StorageType) -> Storage;
    fn train(&mut self, vectors: &[Vec<f32>]) -> Result<(), QuantizationError>;
}

pub enum StorageType {
    UnsignedByte,
    SubByte(u8),
    HalfPrecisionFP,
}

#[derive(Debug)]
pub enum QuantizationError {
    InvalidInput,
    TrainingFailed,
}
