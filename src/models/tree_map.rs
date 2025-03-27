// TODO: remove this attribute
// This module is not used anywhere currently, once it being used, remove this attribute
#![allow(unused)]

use std::sync::{
    atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering},
    Arc, RwLock,
};

use super::{
    atomic_array::AtomicArray, common::TSHashTable, types::FileOffset, utils::calculate_path,
};

pub struct TreeMap<T> {
    pub(crate) root: TreeMapNode<T>,
}

pub struct TreeMapNode<T> {
    pub node_idx: u16,
    pub offset: RwLock<Option<FileOffset>>,
    pub quotients: QuotientsMap<T>,
    pub children: AtomicArray<Self, 8>,
    pub dirty: AtomicBool,
}

pub struct QuotientsMap<T> {
    pub offset: RwLock<Option<FileOffset>>,
    pub map: TSHashTable<u64, Arc<Quotient<T>>>,
    pub len: AtomicU64,
    pub serialized_upto: AtomicUsize,
}

pub struct Quotient<T> {
    pub sequence_idx: u64,
    pub value: AtomicPtr<T>,
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for TreeMapNode<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.offset.read().unwrap() == *other.offset.read().unwrap()
            && self.quotients == other.quotients
            && self.children == other.children
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for TreeMapNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapNode")
            .field("offset", &*self.offset.read().unwrap())
            .field("quotients", &self.quotients)
            .field("children", &self.children)
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for QuotientsMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
            && self.len.load(Ordering::Relaxed) == other.len.load(Ordering::Relaxed)
            && self.serialized_upto.load(Ordering::Relaxed)
                == other.serialized_upto.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for QuotientsMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotientMap")
            .field("map", &self.map)
            .field("len", &self.len.load(Ordering::Relaxed))
            .field(
                "serialized_upto",
                &self.serialized_upto.load(Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for Quotient<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_idx == other.sequence_idx
            && unsafe {
                self.value.load(Ordering::Relaxed).as_ref()
                    == other.value.load(Ordering::Relaxed).as_ref()
            }
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for Quotient<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Quotient")
            .field("sequence_idx", &self.sequence_idx)
            .field("value", unsafe {
                &self.value.load(Ordering::Relaxed).as_ref()
            })
            .finish()
    }
}

impl<T> TreeMapNode<T> {
    pub fn new(node_idx: u16) -> Self {
        Self {
            node_idx,
            offset: RwLock::new(None),
            quotients: QuotientsMap::default(),
            children: AtomicArray::default(),
            dirty: AtomicBool::new(true),
        }
    }

    pub fn find_or_create_node(&self, path: &[u8]) -> &Self {
        let mut current = self;
        for &idx in path {
            let new_node_idx = current.node_idx + (1u16 << (idx * 2));
            let (child, _) = current.children.get_or_insert(idx as usize, || {
                Box::into_raw(Box::new(Self::new(new_node_idx)))
            });
            current = unsafe { &*child };
        }
        current
    }

    pub fn set_data(&self, quotient: u64, value: T) {
        self.quotients.insert(quotient, value);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get_data(&self, quotient: u64) -> Option<&T> {
        self.quotients.get(quotient)
    }
}

impl<T> Default for QuotientsMap<T> {
    fn default() -> Self {
        Self {
            offset: RwLock::new(None),
            map: TSHashTable::new(16),
            len: AtomicU64::new(0),
            serialized_upto: AtomicUsize::new(0),
        }
    }
}

impl<T> QuotientsMap<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, quotient: u64, value: T) {
        let value_ptr = Box::into_raw(Box::new(value));
        self.map.modify_or_insert(
            quotient,
            |quotient| {
                quotient.value.store(value_ptr, Ordering::Release);
            },
            || {
                Arc::new(Quotient {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    value: AtomicPtr::new(value_ptr),
                })
            },
        );
    }

    fn get(&self, quotient: u64) -> Option<&T> {
        self.map
            .lookup(&quotient)
            .and_then(|q| unsafe { q.value.load(Ordering::Acquire).as_ref() })
    }
}

impl<T> Default for TreeMap<T> {
    fn default() -> Self {
        Self {
            root: TreeMapNode::new(0),
        }
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for TreeMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for TreeMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMap").field("root", &self.root).finish()
    }
}

impl<T> TreeMap<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, key: u64, value: T) {
        let node_pos = (key % 65536) as u32; // its small enough to fit inside u16, but u32 is used to match the `calculate_path` function signature
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.set_data(key, value);
    }

    pub fn get(&self, key: u64) -> Option<&T> {
        let node_pos = (key % 65536) as u32; // its small enough to fit inside u16, but u32 is used to match the `calculate_path` function signature
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_data(key)
    }
}
