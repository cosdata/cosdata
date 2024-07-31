use half::f16;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum Storage {
    UnsignedByte {
        mag: u32,
        quant_vec: Vec<u8>,
    },
    SubByte {
        mag: u32,
        quant_vec: Vec<Vec<u32>>,
        resolution: u8,
    },
    HalfPrecisionFP {
        mag: f32,
        quant_vec: Vec<f16>,
    },
}
