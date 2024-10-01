pub mod inverted_index_new;
pub mod inverted_index_old;
pub mod knn_query_new;
pub mod knn_query_old;

use half::f16;
use serde::{Deserialize, Serialize};

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
    PartialEq,
)]
pub enum Storage {
    UnsignedByte {
        mag: u32,
        quant_vec: Vec<u8>,
    },
    SubByte {
        mag: u32,
        quant_vec: Vec<Vec<u8>>,
        resolution: u8,
    },
    HalfPrecisionFP {
        mag: f32,
        quant_vec: Vec<f16>,
    },
}
