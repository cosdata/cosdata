use super::CustomSerialize;
use crate::models::versioning::Hash;
use crate::storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasic;
use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::NodeRegistry,
        lazy_load::FileIndex,
    },
    storage::inverted_index_sparse_ann::InvertedIndexSparseAnnNode,
};
use std::collections::HashSet;
use std::sync::Arc;

impl CustomSerialize for InvertedIndexSparseAnnNode {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        //todo! Implement serialize

        Ok(0u32)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        //todo! Implement deserialize

        Ok(InvertedIndexSparseAnnNode::new(0, false))
    }
}

impl CustomSerialize for InvertedIndexSparseAnnNodeBasic {
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        //todo! Implement serialize

        Ok(0u32)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        //todo! Implement deserialize

        Ok(InvertedIndexSparseAnnNodeBasic::new(0, false))
    }
}
