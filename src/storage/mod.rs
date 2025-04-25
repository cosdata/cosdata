use half::f16;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq,
)]
pub enum Storage {
    UnsignedByte {
        mag: f32,
        quant_vec: Vec<u8>,
    },
    SubByte {
        mag: f32,
        quant_vec: Vec<Vec<u8>>,
        resolution: u8,
    },
    HalfPrecisionFP {
        mag: f32,
        quant_vec: Vec<f16>,
    },
    FullPrecisionFP {
        mag: f32,
        vec: Vec<f32>,
    },
}
