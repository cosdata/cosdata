use std::sync::atomic::AtomicUsize;

use serde::{Deserialize, Serialize};

#[derive(
    Debug,
    Clone,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
    PartialEq,
    Serialize,
    Deserialize,
)]
pub struct SparsePair(pub u32, pub f32);

#[derive(Default)]
pub struct SamplingData {
    pub above_1: AtomicUsize,
    pub above_2: AtomicUsize,
    pub above_3: AtomicUsize,
    pub above_4: AtomicUsize,
    pub above_5: AtomicUsize,
    pub above_6: AtomicUsize,
    pub above_7: AtomicUsize,
    pub above_8: AtomicUsize,
    pub above_9: AtomicUsize,
    pub values_collected: AtomicUsize,
}
