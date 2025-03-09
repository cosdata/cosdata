use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::models::types::{RawVectorEmbedding, VectorId};

// Raw vector embedding
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq)]
pub struct RawSparseVectorEmbedding {
    pub raw_vec: Arc<Vec<SparsePair>>,
    pub hash_vec: VectorId,
}

impl RawSparseVectorEmbedding {
    pub fn into_dense(&self, len: usize) -> RawVectorEmbedding {
        let mut raw_vec = vec![0.0; len];
        for pair in &*self.raw_vec {
            let idx = pair.0 as usize;
            if idx < len {
                raw_vec[idx] = pair.1;
            }
        }

        RawVectorEmbedding {
            raw_vec: Arc::new(raw_vec),
            hash_vec: self.hash_vec.clone(),
        }
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
