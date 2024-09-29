use std::{collections::HashSet, sync::Arc};

use arcshift::ArcShift;

use crate::{
    models::{
        buffered_io::{BufIoError, BufferManagerFactory},
        cache_loader::NodeRegistry,
        lazy_load::FileIndex,
        versioning::Hash,
    },
    storage::inverted_index::{InvertedIndex, InvertedIndexNode},
};

use super::CustomSerialize;

impl<T> CustomSerialize for InvertedIndex<T>
where
    T: Clone + CustomSerialize + 'static,
    InvertedIndexNode<T>: CustomSerialize,
{
    fn serialize(
        &self,
        bufmans: Arc<BufferManagerFactory>,
        version: Hash,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let mut root_arc = self.root.clone();
        root_arc.get().serialize(bufmans, version, cursor)
    }

    fn deserialize(
        bufmans: Arc<BufferManagerFactory>,
        file_index: FileIndex,
        cache: Arc<NodeRegistry>,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<Self, BufIoError> {
        let root =
            InvertedIndexNode::deserialize(bufmans, file_index, cache.clone(), max_loads, skipm)?;
        Ok(Self {
            root: ArcShift::new(root),
            cache,
        })
    }
}
