use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManagerFactory},
    common::TSHashTable,
    serializer::{PartitionedSerialize, SimpleSerialize},
    tf_idf_index::{VersionedVec, VersionedVecIter},
    types::FileOffset,
    utils::calculate_path,
    versioning::VersionNumber,
};

pub trait TreeMapKey: std::hash::Hash + Eq {
    fn key(&self) -> u64;
}

pub struct TreeMap<K, V> {
    pub(crate) root: TreeMapNode<V>,
    bufmans: BufferManagerFactory<u8>,
    _marker: PhantomData<K>,
}

pub struct TreeMapVec<K, V> {
    pub(crate) root: TreeMapVecNode<V>,
    bufmans: BufferManagerFactory<u8>,
    _marker: PhantomData<K>,
}

pub struct TreeMapNode<T> {
    pub node_idx: u16,
    pub offset: RwLock<Option<FileOffset>>,
    pub quotients: QuotientsMap<T>,
    pub children: AtomicArray<Self, 8>,
    pub dirty: AtomicBool,
}

pub struct TreeMapVecNode<T> {
    pub node_idx: u16,
    pub offset: RwLock<Option<FileOffset>>,
    pub quotients: QuotientsMapVec<T>,
    pub children: AtomicArray<Self, 8>,
    pub dirty: AtomicBool,
}

pub struct QuotientsMap<T> {
    pub offset: RwLock<Option<FileOffset>>,
    pub map: TSHashTable<u64, Arc<Quotient<T>>>,
    pub len: AtomicU64,
    pub serialized_upto: AtomicUsize,
}

pub struct QuotientsMapVec<T> {
    pub offset: RwLock<Option<FileOffset>>,
    pub map: TSHashTable<u64, Arc<QuotientVec<T>>>,
    pub len: AtomicU64,
    pub serialized_upto: AtomicUsize,
}

pub struct Quotient<T> {
    pub sequence_idx: u64,
    pub value: VersionedItem<T>,
}

pub struct QuotientVec<T> {
    pub sequence_idx: u64,
    pub value: RwLock<VersionedVec<T>>,
}

pub struct VersionedItem<T> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: VersionNumber,
    pub value: T,
    pub next: RwLock<Option<Box<Self>>>,
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
impl<T: PartialEq> PartialEq for TreeMapVecNode<T> {
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
impl<T: std::fmt::Debug> std::fmt::Debug for TreeMapVecNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapVecNode")
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
impl<T: PartialEq> PartialEq for QuotientsMapVec<T> {
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
impl<T: std::fmt::Debug> std::fmt::Debug for QuotientsMapVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotientMapVec")
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
impl<T: PartialEq> PartialEq for QuotientVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_idx == other.sequence_idx
            && *self.value.read().unwrap() == *other.value.read().unwrap()
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
impl<T: std::fmt::Debug> std::fmt::Debug for QuotientVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotientVec")
            .field("sequence_idx", &self.sequence_idx)
            .field("value", &self.value)
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for VersionedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.value == other.value
            && *self.next.read().unwrap() == *other.next.read().unwrap()
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for VersionedItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeVersionedItem")
            .field("version", &self.version)
            .field("value", &self.value)
            .field("next", &*self.next.read().unwrap())
            .finish()
    }
}

impl<T> VersionedItem<T> {
    pub fn new(version: VersionNumber, value: T) -> Self {
        Self {
            serialized_at: RwLock::new(None),
            version,
            value,
            next: RwLock::new(None),
        }
    }

    pub fn insert(&self, version: VersionNumber, value: T) {
        self.insert_inner(Box::new(Self::new(version, value)));
    }

    fn insert_inner(&self, version: Box<Self>) {
        if let Some(next) = &*self.next.read().unwrap() {
            return next.insert_inner(version);
        }

        let mut next_write_guard = self.next.write().unwrap();

        if let Some(next) = &*next_write_guard {
            return next.insert_inner(version);
        }

        *next_write_guard = Some(version);
    }

    pub fn latest<'a>(&'a self) -> &'a T {
        if let Some(next) = &*self.next.read().unwrap() {
            let next = unsafe { std::mem::transmute::<&Box<Self>, &'a Box<Self>>(next) };
            return next.latest();
        }

        &self.value
    }

    pub fn latest_item<'a>(&'a self) -> &'a Self {
        if let Some(next) = &*self.next.read().unwrap() {
            let next = unsafe { std::mem::transmute::<&Box<Self>, &'a Box<Self>>(next) };
            return next.latest_item();
        }

        self
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

    pub fn insert(&self, version: VersionNumber, quotient: u64, value: T) {
        self.quotients.insert(version, quotient, value);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get_latest(&self, quotient: u64) -> Option<&T> {
        self.quotients.get_latest(quotient)
    }

    pub fn get_versioned(&self, quotient: u64) -> Option<&VersionedItem<T>> {
        self.quotients.get_versioned(quotient)
    }
}

impl<T> TreeMapVecNode<T> {
    pub fn new(node_idx: u16) -> Self {
        Self {
            node_idx,
            offset: RwLock::new(None),
            quotients: QuotientsMapVec::default(),
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

    pub fn push(&self, version: VersionNumber, quotient: u64, value: T) {
        self.quotients.push(version, quotient, value);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get(&self, quotient: u64) -> Option<VersionedVecIter<'_, T>> {
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

impl<T> Default for QuotientsMapVec<T> {
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
    pub fn insert(&self, version: VersionNumber, quotient: u64, value: T) {
        self.map.modify_or_insert_with_value(
            quotient,
            value,
            |value, quotient| {
                quotient.value.insert(version, value);
            },
            |value| {
                Arc::new(Quotient {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    value: VersionedItem::new(version, value),
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

    fn get_versioned(&self, quotient: u64) -> Option<&VersionedItem<T>> {
        self.map.lookup(&quotient).map(|q| {
            // SAFETY: here we are changing the lifetime of the value by using
            // `std::mem::transmute`, which by definition is not safe, but given our use of
            // this value and how we destroy it, its actually safe in this context.
            unsafe { std::mem::transmute(&q.value) }
        })
    }
}

impl<T> QuotientsMapVec<T> {
    pub fn push(&self, version: VersionNumber, quotient: u64, value: T) {
        self.map.modify_or_insert_with_value(
            quotient,
            value,
            |value, quotient| {
                quotient.value.write().unwrap().push(version, value);
            },
            |value| {
                let mut vec = VersionedVec::new(version);
                vec.push(version, value);
                Arc::new(QuotientVec {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    value: RwLock::new(vec),
                })
            },
        );
    }

    fn get<'a>(&self, quotient: u64) -> Option<VersionedVecIter<'a, T>> {
        self.map.lookup(&quotient).map(|q| {
            unsafe {
                std::mem::transmute::<&VersionedVec<T>, &'a VersionedVec<T>>(
                    &*q.value.read().unwrap(),
                )
            }
            .iter()
        })
    }
}

#[cfg(test)]
impl<K: TreeMapKey + Clone, V: PartialEq> PartialEq for TreeMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

#[cfg(test)]
impl<K: TreeMapKey + Clone, V: PartialEq> PartialEq for TreeMapVec<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

#[cfg(test)]
impl<K: std::fmt::Debug + TreeMapKey + Clone + Ord, V: std::fmt::Debug> std::fmt::Debug
    for TreeMap<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMap").field("root", &self.root).finish()
    }
}

#[cfg(test)]
impl<K: std::fmt::Debug + TreeMapKey + Clone + Ord, V: std::fmt::Debug> std::fmt::Debug
    for TreeMapVec<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapVec")
            .field("root", &self.root)
            .finish()
    }
}

impl<K: TreeMapKey, V> TreeMap<K, V> {
    pub fn new(bufmans: BufferManagerFactory<u8>) -> Self {
        Self {
            root: TreeMapNode::new(0),
            bufmans,
            _marker: PhantomData,
        }
    }
}

impl<K: TreeMapKey, V> TreeMap<K, V> {
    pub fn insert(&self, version: VersionNumber, key: &K, value: V) {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.insert(version, key, value);
    }

    pub fn get_latest(&self, key: &K) -> Option<&V> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_latest(key)
    }

    pub fn get_versioned(&self, key: &K) -> Option<&VersionedItem<V>> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_versioned(key)
    }
}

impl<K, V: SimpleSerialize> TreeMap<K, V> {
    pub fn serialize(&self, file_parts: u8) -> Result<(), BufIoError> {
        let bufman = self.bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let offset = self.root.serialize(&self.bufmans, file_parts, 0, 0)?;
        bufman.seek_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, offset)?;
        bufman.close_cursor(cursor)?;
        self.bufmans.flush_all()?;
        Ok(())
    }

    pub fn deserialize(
        bufmans: BufferManagerFactory<u8>,
        file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let bufman = bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        let offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(Self {
            root: TreeMapNode::deserialize(&bufmans, file_parts, 0, FileOffset(offset))?,
            bufmans,
            _marker: PhantomData,
        })
    }
}

impl<K: TreeMapKey, V> TreeMapVec<K, V> {
    pub fn new(bufmans: BufferManagerFactory<u8>) -> Self {
        Self {
            root: TreeMapVecNode::new(0),
            bufmans,
            _marker: PhantomData,
        }
    }
}

impl<K: TreeMapKey, V> TreeMapVec<K, V> {
    pub fn push(&self, version: VersionNumber, key: &K, value: V) {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.push(version, key, value);
    }

    pub fn get<'a>(&'a self, key: &K) -> Option<VersionedVecIter<'a, V>> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get(key)
    }
}

impl<K, V: SimpleSerialize> TreeMapVec<K, V> {
    pub fn serialize(&self, file_parts: u8) -> Result<(), BufIoError> {
        let bufman = self.bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let offset = self.root.serialize(&self.bufmans, file_parts, 0, 0)?;
        bufman.seek_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, offset)?;
        bufman.close_cursor(cursor)?;
        self.bufmans.flush_all()?;
        Ok(())
    }

    pub fn deserialize(
        bufmans: BufferManagerFactory<u8>,
        file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let bufman = bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        let offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(Self {
            root: TreeMapVecNode::deserialize(&bufmans, file_parts, 0, FileOffset(offset))?,
            bufmans,
            _marker: PhantomData,
        })
    }
}

impl TreeMapKey for u64 {
    fn key(&self) -> u64 {
        *self
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn tests_basic_usage() {
        let tempdir = tempdir().unwrap();
        let bufmans = BufferManagerFactory::new(
            tempdir.as_ref().into(),
            |root, part| root.join(format!("{}.tree-map", part)),
            8192,
        );
        let map: TreeMap<u64, u64> = TreeMap::new(bufmans);
        map.insert(0.into(), &65536, 0);
        map.insert(0.into(), &0, 23);
        assert_eq!(map.get_latest(&0), Some(&23));
        map.insert(1.into(), &0, 29);
        assert_eq!(map.get_latest(&0), Some(&29));
        assert_eq!(
            map.get_versioned(&0),
            Some(&VersionedItem {
                version: 0.into(),
                serialized_at: RwLock::new(None),
                value: 23,
                next: RwLock::new(Some(Box::new(VersionedItem::new(1.into(), 29))))
            })
        );
    }
}
