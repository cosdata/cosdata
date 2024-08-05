use arcshift::ArcShift;

use super::types::Item;
use std::sync::{Arc, RwLock};

pub trait SyncPersist {
    fn set_persistence(&self, flag: bool);
    fn needs_persistence(&self) -> bool;
}

type FileOffset = u32;
type VersionId = u16;
type HNSWLevel = u8;

pub const CHUNK_SIZE: usize = 5;

// #[derive(Debug, Clone)]
#[derive(Clone)]
pub enum LazyItem<T: Clone + 'static> {
    Valid {
        data: Option<Item<T>>,
        offset: Option<FileOffset>,
        decay_counter: usize,
    },
    Invalid,
}

// #[derive(Debug, Clone)]
#[derive(Clone)]
pub struct LazyItemRef<T: Clone + 'static> {
    pub item: Item<LazyItem<T>>,
}

// #[derive(Debug, Clone)]
#[derive(Clone)]
pub struct LazyItems<T: Clone + 'static> {
    pub items: Item<Vec<LazyItemRef<T>>>,
}

impl<T: Clone> LazyItem<T> {
    pub fn with_data(data: T) -> Self {
        LazyItem::Valid {
            data: Some(Item::new(data)),
            offset: None,
            decay_counter: 0,
        }
    }
}

impl<T: Clone> LazyItemRef<T> {
    pub fn new(item: T) -> Self {
        LazyItemRef {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(ArcShift::new(item)),
                offset: None,
                decay_counter: 0,
            }),
        }
    }

    pub fn from_item(item: Item<T>) -> Self {
        LazyItemRef {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(item),
                offset: None,
                decay_counter: 0,
            }),
        }
    }

    pub fn get_data(&mut self) -> Option<Item<T>> {
        if let LazyItem::Valid { data, .. } = self.item.get() {
            return data.clone();
        }
        None
    }

    pub fn set_data(&mut self, new_data: T) {
        self.item.rcu(|item| {
            let (offset, decay_counter) = if let LazyItem::Valid {
                offset,
                decay_counter,
                ..
            } = item
            {
                (offset.clone(), *decay_counter)
            } else {
                (None, 0)
            };
            LazyItem::Valid {
                data: Some(Item::new(new_data)),
                offset,
                decay_counter,
            }
        });
    }

    pub fn set_offset(&mut self, new_offset: Option<FileOffset>) {
        self.item.rcu(|item| {
            let (data, decay_counter) = if let LazyItem::Valid {
                data,
                decay_counter,
                ..
            } = item
            {
                (data.clone(), *decay_counter)
            } else {
                (None, 0)
            };
            LazyItem::Valid {
                data,
                offset: new_offset,
                decay_counter,
            }
        });
    }
}

impl<T: Clone> LazyItems<T> {
    pub fn new() -> Self {
        LazyItems {
            items: Item::new(Vec::new()),
        }
    }

    pub fn push(&mut self, item: LazyItemRef<T>) {
        self.items.rcu(|items| {
            let mut items = items.clone();
            items.push(item);
            items
        });
    }

    pub fn get(&mut self, index: usize) -> Option<LazyItemRef<T>> {
        self.items.get().get(index).cloned()
    }

    pub fn len(&mut self) -> usize {
        self.items.get().len()
    }

    pub fn is_empty(&mut self) -> bool {
        self.items.get().is_empty()
    }

    pub fn iter(&mut self) -> impl Iterator<Item = LazyItemRef<T>> {
        self.items.get().clone().into_iter()
    }
}
