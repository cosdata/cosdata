use std::collections::HashSet;

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::ProbCache,
        lazy_load::FileIndex,
        versioning::Hash,
    },
    storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasicTSHashmap,
};

use super::ProbSerialize;

impl ProbSerialize for InvertedIndexSparseAnnNodeBasicTSHashmap {
    fn serialize(
        &self,
        bufmans: &BufferManagerFactory<Hash>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        todo!()
    }

    fn deserialize(
        bufmans: &BufferManagerFactory<Hash>,
        file_index: FileIndex,
        cache: &ProbCache,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        todo!()
    }
}
