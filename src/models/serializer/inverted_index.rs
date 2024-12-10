use std::{collections::HashSet, sync::Arc};

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::NodeRegistry,
        lazy_load::FileIndex,
        versioning::Hash,
    },
    storage::inverted_index_old::{InvertedIndex, InvertedIndexItem},
};

use super::CustomSerialize;

impl<T> CustomSerialize for InvertedIndex<T>
where
    T: Clone + CustomSerialize + 'static,
    InvertedIndexItem<T>: CustomSerialize,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory<Hash>>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        self.root.serialize(bufmans, version, cursor)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let root =
            InvertedIndexItem::deserialize(bufmans, file_index, cache.clone(), max_loads, skipm)?;
        Ok(Self {
            root: Arc::new(root),
            cache,
        })
    }
}
