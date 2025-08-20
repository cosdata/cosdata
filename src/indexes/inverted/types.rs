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
