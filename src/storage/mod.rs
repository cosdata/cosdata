pub mod bench_common;
pub mod inverted_index_old;
pub mod inverted_index_sparse_ann;
pub mod inverted_index_sparse_ann_basic;
pub mod inverted_index_sparse_ann_new_ds;
pub mod knn_query_old;
pub mod sparse_ann_query;
pub mod sparse_ann_query_basic;
pub mod sparse_ann_query_new_ds;


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
