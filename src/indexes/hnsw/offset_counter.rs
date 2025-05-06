use crate::models::{prob_node::ProbNode, types::FileOffset};
use serde::{Deserialize, Serialize};
use std::{
    ops::Deref,
    sync::atomic::{AtomicU32, Ordering},
};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFileId(u32);

impl From<u32> for IndexFileId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl Deref for IndexFileId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl IndexFileId {
    pub fn invalid() -> Self {
        Self(u32::MAX)
    }
}

pub struct HNSWIndexFileOffsetCounter {
    pub offset: AtomicU32,
    pub level_0_node_size: u32,
    pub node_size: u32,
    pub min_size: u32,
    pub file_id: u32,
}

impl HNSWIndexFileOffsetCounter {
    pub fn new(min_size: u32, level_0_neighbors_count: usize, neighbours_count: usize) -> Self {
        Self {
            offset: AtomicU32::new(0),
            level_0_node_size: ProbNode::get_serialized_size(level_0_neighbors_count) as u32,
            node_size: ProbNode::get_serialized_size(neighbours_count) as u32,
            min_size,
            file_id: 0,
        }
    }

    pub fn from_offset_and_file_id(
        offset: u32,
        file_id: u32,
        min_size: u32,
        level_0_neighbors_count: usize,
        neighbours_count: usize,
    ) -> Self {
        Self {
            offset: AtomicU32::new(offset),
            level_0_node_size: ProbNode::get_serialized_size(level_0_neighbors_count) as u32,
            node_size: ProbNode::get_serialized_size(neighbours_count) as u32,
            min_size,
            file_id,
        }
    }

    pub fn next_file_id(&mut self) -> IndexFileId {
        if self.offset.load(Ordering::Relaxed) >= self.min_size {
            self.offset.store(0, Ordering::Relaxed);
            self.file_id += 1;
        }

        IndexFileId::from(self.file_id)
    }

    pub fn file_id(&self) -> IndexFileId {
        IndexFileId::from(self.file_id)
    }

    pub fn next_level_0_offset(&self) -> FileOffset {
        FileOffset(
            self.offset
                .fetch_add(self.level_0_node_size, Ordering::Relaxed),
        )
    }

    pub fn next_offset(&self) -> FileOffset {
        FileOffset(self.offset.fetch_add(self.node_size, Ordering::Relaxed))
    }
}
