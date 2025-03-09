use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::models::types::VectorId;

// Raw vector embedding
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq)]
pub struct RawSparseVectorEmbedding {
    pub raw_vec: Arc<Vec<SparsePair>>,
    pub hash_vec: VectorId,
}

impl RawSparseVectorEmbedding {
    pub fn into_map(&self) -> FxHashMap<u32, f32> {
        let mut map = FxHashMap::default();

        for pair in &*self.raw_vec {
            map.insert(pair.0, pair.1);
        }
        map
    }
}

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
