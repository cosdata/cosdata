use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, RwLock};

pub trait SyncPersist {
    fn set_persistence(&self, flag: bool);
    fn needs_persistence(&self) -> bool;
}

type FileOffset = u32;
type VersionId = u16;
type HNSWLevel = u8;

pub const CHUNK_SIZE: usize = 5;

#[derive(Debug, Clone)]
pub enum LazyItem<T: Clone> {
    Valid {
        data: Option<Arc<T>>,
        offset: Option<FileOffset>,
        decay_counter: usize,
    },
    Invalid,
}

pub type LazyItemRef<T: Clone> = Arc<RwLock<LazyItem<T>>>;

#[derive(Debug, Clone)]
pub struct LazyItems<T: Clone> {
    pub items: Arc<RwLock<Vec<LazyItemRef<T>>>>,
}

impl<T: Clone> LazyItem<T> {
    pub fn new() -> Self {
        LazyItem::Valid {
            data: None,
            offset: None,
            decay_counter: 0,
        }
    }

    pub fn with_data(data: T) -> Self {
        LazyItem::Valid {
            data: Some(Arc::new(data)),
            offset: None,
            decay_counter: 0,
        }
    }

    pub fn get_offset(&self) -> Option<FileOffset> {
        match self {
            LazyItem::Valid { offset, .. } => *offset,
            LazyItem::Invalid => None,
        }
    }

    pub fn set_offset(&mut self, new_offset: Option<FileOffset>) {
        if let LazyItem::Valid { offset, .. } = self {
            *offset = new_offset;
        }
    }
    pub fn is_valid(&self) -> bool {
        matches!(self, LazyItem::Valid { .. })
    }

    pub fn get_data(&self) -> Option<Arc<T>> {
        match self {
            LazyItem::Valid { data, .. } => data.clone(),
            LazyItem::Invalid => None,
        }
    }

    pub fn set_data(&mut self, new_data: Option<Arc<T>>) {
        if let LazyItem::Valid { data, .. } = self {
            *data = new_data;
        }
    }

    pub fn increment_decay(&mut self) {
        if let LazyItem::Valid { decay_counter, .. } = self {
            *decay_counter += 1;
        }
    }

    pub fn reset_decay(&mut self) {
        if let LazyItem::Valid { decay_counter, .. } = self {
            *decay_counter = 0;
        }
    }
}

impl<T: Clone> LazyItems<T> {
    pub fn new() -> Self {
        LazyItems {
            items: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn push(&self, item: LazyItemRef<T>) {
        self.items.write().unwrap().push(item);
    }

    pub fn get(&self, index: usize) -> Option<LazyItemRef<T>> {
        self.items.read().unwrap().get(index).cloned()
    }

    pub fn len(&self) -> usize {
        self.items.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.read().unwrap().is_empty()
    }

    pub fn iter(&self) -> Vec<LazyItemRef<T>> {
        self.items.read().unwrap().clone()
    }
}
