// TODO: remove this attribute
// This module is not used anywhere currently, once it being used, remove this attribute
#![allow(unused)]

use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManagerFactory},
    common::TSHashTable,
    serializer::{PartitionedSerialize, SimpleSerialize},
    types::FileOffset,
    utils::calculate_path,
    versioning::Hash,
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
    pub value: UnsafeVersionedItem<T>,
}

pub struct UnsafeVersionedItem<T> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: Hash,
    pub value: T,
    pub next: UnsafeCell<Option<Box<Self>>>,
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
        self.sequence_idx == other.sequence_idx && self.value == other.value
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for Quotient<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Quotient")
            .field("sequence_idx", &self.sequence_idx)
            .field("value", &self.value)
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for UnsafeVersionedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.value == other.value
            && unsafe { *self.next.get() == *other.next.get() }
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for UnsafeVersionedItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeVersionedItem")
            .field("version", &self.version)
            .field("value", &self.value)
            .field("next", unsafe { &*self.next.get() })
            .finish()
    }
}

impl<T> UnsafeVersionedItem<T> {
    pub fn new(version: Hash, value: T) -> Self {
        Self {
            serialized_at: RwLock::new(None),
            version,
            value,
            next: UnsafeCell::new(None),
        }
    }

    pub fn push(&self, version: Hash, value: T) {
        self.push_inner(Box::new(Self::new(version, value)));
    }

    fn push_inner(&self, version: Box<Self>) {
        if let Some(next) = unsafe { &*self.next.get() } {
            return next.push_inner(version);
        }

        unsafe {
            *self.next.get() = Some(version);
        }
    }

    pub fn latest(&self) -> &T {
        if let Some(next) = unsafe { &*self.next.get() } {
            return next.latest();
        }

        &self.value
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

    pub fn insert(&self, version: Hash, quotient: u64, value: T) {
        self.quotients.insert(version, quotient, value);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get_latest(&self, quotient: u64) -> Option<&T> {
        self.quotients.get_latest(quotient)
    }

    pub fn get_versioned(&self, quotient: u64) -> Option<&UnsafeVersionedItem<T>> {
        self.quotients.get_versioned(quotient)
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

    pub fn insert(&self, version: Hash, quotient: u64, value: T) {
        self.map.modify_or_insert_with_value(
            quotient,
            value,
            |value, quotient| {
                quotient.value.push(version, value);
            },
            |value| {
                Arc::new(Quotient {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    value: UnsafeVersionedItem::new(version, value),
                })
            },
        );
    }

    fn get_latest(&self, quotient: u64) -> Option<&T> {
        self.map.lookup(&quotient).map(|q| {
            // SAFETY: here we are changing the lifetime of the value by using
            // `std::mem::transmute`, which by definition is not safe, but given our use of
            // this value and how we destroy it, its actually safe in this context.
            unsafe { std::mem::transmute(q.value.latest()) }
        })
    }

    fn get_versioned(&self, quotient: u64) -> Option<&UnsafeVersionedItem<T>> {
        self.map.lookup(&quotient).map(|q| {
            // SAFETY: here we are changing the lifetime of the value by using
            // `std::mem::transmute`, which by definition is not safe, but given our use of
            // this value and how we destroy it, its actually safe in this context.
            unsafe { std::mem::transmute(&q.value) }
        })
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

    pub fn insert(&self, version: Hash, key: u64, value: T) {
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.insert(version, key, value);
    }

    pub fn get_latest(&self, key: u64) -> Option<&T> {
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_latest(key)
    }

    pub fn get_versioned(&self, key: u64) -> Option<&UnsafeVersionedItem<T>> {
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_versioned(key)
    }
}

impl<T: SimpleSerialize> TreeMap<T> {
    pub fn serialize(
        &self,
        bufmans: &BufferManagerFactory<u8>,
        file_parts: u8,
    ) -> Result<(), BufIoError> {
        let bufman = bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let offset = self.root.serialize(bufmans, file_parts, 0, 0)?;
        bufman.seek_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, offset)?;
        bufman.close_cursor(cursor)?;
        Ok(())
    }

    pub fn deserialize(
        bufmans: &BufferManagerFactory<u8>,
        file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let bufman = bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        let offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(Self {
            root: TreeMapNode::deserialize(bufmans, file_parts, 0, FileOffset(offset))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tests_basic_usage() {
        let map = TreeMap::new();
        map.insert(0.into(), 65536, 0);
        map.insert(0.into(), 0, 23);
        assert_eq!(map.get_latest(0), Some(&23));
        map.insert(1.into(), 0, 29);
        assert_eq!(map.get_latest(0), Some(&29));
        assert_eq!(
            map.get_versioned(0),
            Some(&UnsafeVersionedItem {
                version: 0.into(),
                serialized_at: RwLock::new(None),
                value: 23,
                next: UnsafeCell::new(Some(Box::new(UnsafeVersionedItem::new(1.into(), 29))))
            })
        );
    }
}
