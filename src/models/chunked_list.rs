use super::types::{FileOffset, Item};
use std::sync::{Arc, RwLock};

pub trait SyncPersist {
    fn set_persistence(&self, flag: bool);
    fn needs_persistence(&self) -> bool;
}

pub const CHUNK_SIZE: usize = 5;

#[derive(Debug, Clone)]
pub struct LazyItem<T: Clone> {
    pub data: Option<Item<T>>,
    pub offset: Option<FileOffset>,
    pub decay_counter: usize,
}

#[derive(Debug, Clone)]
pub struct LazyItemRef<T: Clone> {
    pub item: Item<LazyItem<T>>,
}

#[derive(Debug, Clone)]
pub struct LazyItems<T: Clone> {
    pub items: Item<Vec<LazyItem<T>>>,
}

impl<T: Clone> LazyItem<T> {
    pub fn with_data(data: T) -> Self {
        LazyItem {
            data: Some(Arc::new(RwLock::new(data))),
            offset: None,
            decay_counter: 0,
        }
    }
}

impl<T: Clone> LazyItemRef<T> {
    pub fn new(item: T) -> Self {
        LazyItemRef {
            item: Arc::new(RwLock::new(LazyItem {
                data: Some(Arc::new(RwLock::new(item))),
                offset: None,
                decay_counter: 0,
            })),
        }
    }

    pub fn new_with_lock(item: Item<T>) -> Self {
        LazyItemRef {
            item: Arc::new(RwLock::new(LazyItem {
                data: Some(item),
                offset: None,
                decay_counter: 0,
            })),
        }
    }

    pub fn get_data(&self) -> Option<Item<T>> {
        self.item.read().unwrap().data.clone()
    }

    pub fn set_data(&self, new_data: T) {
        self.item.write().unwrap().data = Some(Arc::new(RwLock::new(new_data)));
    }

    pub fn set_offset(&self, new_offset: Option<FileOffset>) {
        self.item.write().unwrap().offset = new_offset;
    }
}

impl<T: Clone> LazyItems<T> {
    pub fn new() -> Self {
        LazyItems {
            items: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn push(&self, item: LazyItem<T>) {
        self.items.write().unwrap().push(item);
    }

    pub fn get(&self, index: usize) -> Option<LazyItem<T>> {
        self.items.read().unwrap().get(index).cloned()
    }

    pub fn len(&self) -> usize {
        self.items.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.read().unwrap().is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = LazyItem<T>> {
        self.items.read().unwrap().clone().into_iter()
    }
}
