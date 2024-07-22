use std::sync::{Arc, RwLock};

pub trait Locatable {
    fn get_location(&self) -> Option<u32>;
    fn set_location(&mut self, location: u32);
    fn mark_for_persistence(&mut self);
    fn needs_persistence(&self) -> bool;
}

#[derive(Debug, Clone)]
pub enum ItemListRef<T: Clone + Locatable> {
    Ref(Box<Items<T>>),
    Invalid,
}

#[derive(Debug, Clone)]
pub struct Items<T: Clone + Locatable> {
    pub items: [ItemRef<T>; 4],
    pub next: ItemListRef<T>,
}

#[derive(Debug, Clone)]
pub enum ItemRef<T: Clone + Locatable> {
    InMemory(Arc<T>),
    Persistent(FileOffset),
    Invalid,
}

pub type FileOffset = u32;

impl<T: Clone + Locatable> Items<T> {
    pub fn new() -> Self {
        Items {
            items: [
                ItemRef::Invalid,
                ItemRef::Invalid,
                ItemRef::Invalid,
                ItemRef::Invalid,
            ],
            next: ItemListRef::Invalid,
        }
    }

    pub fn add(&mut self, item: ItemRef<T>) {
        for i in 0..self.items.len() {
            if matches!(self.items[i], ItemRef::Invalid) {
                self.items[i] = item;
                return;
            }
        }

        match &mut self.next {
            ItemListRef::Invalid => {
                self.next = ItemListRef::Ref(Box::new(Items::new()));
            }
            ItemListRef::Ref(next_items) => {
                next_items.add(item);
            }
        }
    }

    pub fn get_all(&self) -> Vec<ItemRef<T>> {
        let mut result = Vec::new();
        self.collect_items(&mut result);
        result
    }

    fn collect_items(&self, result: &mut Vec<ItemRef<T>>) {
        for item in &self.items {
            result.push(item.clone());
        }

        if let ItemListRef::Ref(next_items) = &self.next {
            next_items.collect_items(result);
        }
    }

    pub fn set_all(&mut self, new_items: Vec<ItemRef<T>>) {
        let mut item_iter = new_items.into_iter();
        self.set_items(&mut item_iter);
    }

    fn set_items(&mut self, item_iter: &mut std::vec::IntoIter<ItemRef<T>>) {
        for i in 0..self.items.len() {
            self.items[i] = item_iter.next().unwrap_or(ItemRef::Invalid);
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
}

impl<T: Default + Clone + Locatable> Default for ItemListRef<T> {
    fn default() -> Self {
        ItemListRef::Invalid
    }
}

impl<T: Clone + Locatable> ItemListRef<T> {
    pub fn new() -> Self {
        ItemListRef::Ref(Box::new(Items::new()))
    }

    pub fn add(&mut self, item: ItemRef<T>) {
        match self {
            ItemListRef::Ref(items) => items.add(item),
            ItemListRef::Invalid => {
                *self = ItemListRef::Ref(Box::new(Items::new()));
                if let ItemListRef::Ref(items) = self {
                    items.add(item);
                }
            }
        }
    }

    pub fn get_all(&self) -> Vec<ItemRef<T>> {
        match self {
            ItemListRef::Ref(items) => items.get_all(),
            ItemListRef::Invalid => Vec::new(),
        }
    }

    pub fn set_all(&mut self, new_items: Vec<ItemRef<T>>) {
        match self {
            ItemListRef::Ref(items) => items.set_all(new_items),
            ItemListRef::Invalid => {
                if !new_items.is_empty() {
                    *self = ItemListRef::Ref(Box::new(Items::new()));
                    if let ItemListRef::Ref(items) = self {
                        items.set_all(new_items);
                    }
                }
            }
        }
    }
}
