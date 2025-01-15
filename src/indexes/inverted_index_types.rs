use std::sync::Arc;

use serde::Serialize;

use crate::models::types::VectorId;

// Raw vector embedding
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq)]
pub struct RawSparseVectorEmbedding {
    pub raw_vec: Arc<Vec<SparsePair>>,
    pub hash_vec: VectorId,
}

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq, Serialize)]
pub struct SparsePair(pub u32, pub f32);
