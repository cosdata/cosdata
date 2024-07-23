use std::sync::{Arc, RwLock};

pub trait Locatable {
    fn get_file_offset(&self) -> Option<u32>;
    fn set_file_offset(&mut self, location: u32);
    fn set_persistence(&mut self, flag: bool);
    fn needs_persistence(&self) -> bool;
    // fn include_items(&self) -> bool;
    // fn set_include_items(&mut self, value: bool);
}

#[derive(Debug, Clone)]
pub enum ItemListRef<T: Clone + Locatable> {
    Ref(Box<Items<T>>),
    Null,
}

#[derive(Debug, Clone)]
pub struct Items<T: Clone + Locatable> {
    pub items: [LazyItem<T>; 4],
    pub next: ItemListRef<T>,
}

#[derive(Debug, Clone)]
pub enum LazyItem<T: Clone + Locatable> {
    Ready(Arc<T>),
    LazyLoad(FileOffset),
    Null,
}

pub type FileOffset = u32;

impl<T: Clone + Locatable> LazyItem<T> {
    fn get_offset(&self) -> Option<u32> {
        match self {
            LazyItem::LazyLoad(offset) => Some(*offset),
            LazyItem::Ready(item) => item.get_file_offset(),
            LazyItem::Null => None,
        }
    }
}

impl<T: Clone + Locatable> LazyItem<T> {
    pub fn new_lazy(offset: FileOffset) -> Self {
        if offset == u32::MAX {
            LazyItem::Null
        } else {
            LazyItem::LazyLoad(offset)
        }
    }
}

impl<T: Clone + Locatable> Items<T> {
    pub fn new() -> Self {
        Items {
            items: [
                LazyItem::Null,
                LazyItem::Null,
                LazyItem::Null,
                LazyItem::Null,
            ],
            next: ItemListRef::Null,
        }
    }

    pub fn add(&mut self, item: LazyItem<T>) {
        for i in 0..self.items.len() {
            if matches!(self.items[i], LazyItem::Null) {
                self.items[i] = item;
                return;
            }
        }

        match &mut self.next {
            ItemListRef::Null => {
                self.next = ItemListRef::Ref(Box::new(Items::new()));
            }
            ItemListRef::Ref(next_items) => {
                next_items.add(item);
            }
        }
    }

    pub fn get_all(&self) -> Vec<LazyItem<T>> {
        let mut result = Vec::new();
        self.collect_items(&mut result);
        result
    }

    fn collect_items(&self, result: &mut Vec<LazyItem<T>>) {
        for item in &self.items {
            result.push(item.clone());
        }

        if let ItemListRef::Ref(next_items) = &self.next {
            next_items.collect_items(result);
        }
    }

    pub fn set_all(&mut self, new_items: Vec<LazyItem<T>>) {
        let mut item_iter = new_items.into_iter();
        self.set_items(&mut item_iter);
    }

    fn set_items(&mut self, item_iter: &mut std::vec::IntoIter<LazyItem<T>>) {
        for i in 0..self.items.len() {
            self.items[i] = item_iter.next().unwrap_or(LazyItem::Null);
        }

        if let ItemListRef::Ref(next_items) = &mut self.next {
            next_items.set_items(item_iter);
        } else if item_iter.len() > 0 {
            self.next = ItemListRef::Ref(Box::new(Items::new()));
            if let ItemListRef::Ref(next_items) = &mut self.next {
                next_items.set_items(item_iter);
            }
        }
    }

    pub fn mark_all_persistence(&mut self, flag: bool) {
        for item in &mut self.items {
            if let LazyItem::Ready(arc) = item {
                if let Some(inner) = Arc::get_mut(arc) {
                    inner.set_persistence(flag);
                }
            }
        }
        if let ItemListRef::Ref(next_items) = &mut self.next {
            next_items.mark_all_persistence(flag);
        }
    }
}

impl<T: Default + Clone + Locatable> Default for ItemListRef<T> {
    fn default() -> Self {
        ItemListRef::Null
    }
}

impl<T: Clone + Locatable> ItemListRef<T> {
    pub fn new() -> Self {
        ItemListRef::Ref(Box::new(Items::new()))
    }

    pub fn add(&mut self, item: LazyItem<T>) {
        match self {
            ItemListRef::Ref(items) => items.add(item),
            ItemListRef::Null => {
                *self = ItemListRef::Ref(Box::new(Items::new()));
                if let ItemListRef::Ref(items) = self {
                    items.add(item);
                }
            }
        }
    }

    pub fn get_all(&self) -> Vec<LazyItem<T>> {
        match self {
            ItemListRef::Ref(items) => items.get_all(),
            ItemListRef::Null => Vec::new(),
        }
    }

    pub fn set_all(&mut self, new_items: Vec<LazyItem<T>>) {
        match self {
            ItemListRef::Ref(items) => items.set_all(new_items),
            ItemListRef::Null => {
                if !new_items.is_empty() {
                    *self = ItemListRef::Ref(Box::new(Items::new()));
                    if let ItemListRef::Ref(items) = self {
                        items.set_all(new_items);
                    }
                }
            }
        }
    }

    pub fn mark_all_persistence(&mut self, flag: bool) {
        if let ItemListRef::Ref(items) = self {
            items.mark_all_persistence(flag);
        }
    }
}
