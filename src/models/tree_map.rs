use std::{
    collections::VecDeque, hash::{Hash, Hasher}, marker::PhantomData, sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use parking_lot::{RwLock, RwLockReadGuard};
use siphasher::sip::SipHasher24;

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    common::TSHashTable,
    serializer::{tree_map::TreeMapSerialize, SimpleSerialize},
    types::FileOffset,
    utils::calculate_path,
    versioned_vec::{VersionedVec, VersionedVecItem},
    versioning::VersionNumber,
};

pub trait TreeMapKey: std::hash::Hash + Eq {
    fn key(&self) -> u64;
}

impl TreeMapKey for u32 {
    fn key(&self) -> u64 {
        *self as u64
    }
}

// TreeMap is a BufferManager-backed key-value store for internal use
// in cosdata.
pub struct TreeMap<K, V> {
    pub(crate) root: TreeMapNode<V>,
    pub(crate) dim_bufman: BufferManager,
    pub(crate) data_bufmans: BufferManagerFactory<VersionNumber>,
    _marker: PhantomData<K>,
}

pub struct TreeMapVec<K, V> {
    pub(crate) root: TreeMapVecNode<V>,
    pub(crate) dim_bufman: BufferManager,
    pub(crate) data_bufmans: BufferManagerFactory<VersionNumber>,
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
    pub value: RwLock<VersionedItem<T>>,
}

pub struct QuotientVec<T> {
    pub sequence_idx: u64,
    pub value: RwLock<VersionedVec<T>>,
}

pub struct VersionedItem<T> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: VersionNumber,
    pub value: Option<T>,
    pub next: Option<Box<Self>>,
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for TreeMapNode<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.offset.read() == *other.offset.read()
            && self.quotients == other.quotients
            && self.children == other.children
    }
}

#[cfg(test)]
impl<T> PartialEq for TreeMapVecNode<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.offset.read() == *other.offset.read()
            && self.quotients == other.quotients
            && self.children == other.children
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for TreeMapNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapNode")
            .field("offset", &*self.offset.read())
            .field("quotients", &self.quotients)
            .field("children", &self.children)
            .finish()
    }
}

#[cfg(test)]
impl<T> std::fmt::Debug for TreeMapVecNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapVecNode")
            .field("offset", &*self.offset.read())
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
impl<T> PartialEq for QuotientsMapVec<T> {
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
impl<T> std::fmt::Debug for QuotientsMapVec<T> {
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
        self.sequence_idx == other.sequence_idx && *self.value.read() == *other.value.read()
    }
}

#[cfg(test)]
impl<T> PartialEq for QuotientVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_idx == other.sequence_idx && *self.value.read() == *other.value.read()
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
impl<T> std::fmt::Debug for QuotientVec<T> {
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
        self.version == other.version && self.value == other.value && self.next == other.next
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for VersionedItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeVersionedItem")
            .field("version", &self.version)
            .field("value", &self.value)
            .field("next", &self.next)
            .finish()
    }
}

impl<T> VersionedItem<T> {
    pub fn new(version: VersionNumber, value: T) -> Self {
        Self {
            serialized_at: RwLock::new(None),
            version,
            value: Some(value),
            next: None,
        }
    }

    fn new_delete(version: VersionNumber) -> Self {
        Self {
            serialized_at: RwLock::new(None),
            version,
            value: None,
            next: None,
        }
    }

    pub fn insert(&mut self, version: VersionNumber, value: T) {
        self.insert_inner(Box::new(Self::new(version, value)));
    }

    pub fn delete(&mut self, version: VersionNumber) {
        self.insert_inner(Box::new(Self::new_delete(version)));
    }

    fn insert_inner(&mut self, version: Box<Self>) {
        if let Some(next) = &mut self.next {
            return next.insert_inner(version);
        }

        self.next = Some(version);
    }

    pub fn latest(&self) -> Option<&T> {
        if let Some(next) = &self.next {
            return next.latest();
        }

        self.value.as_ref()
    }

    pub fn latest_item(&self) -> &Self {
        if let Some(next) = &self.next {
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

    pub fn delete(&self, version: VersionNumber, quotient: u64) {
        self.quotients.delete(version, quotient);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get_latest(&self, quotient: u64) -> Option<&T> {
        self.quotients.get_latest(quotient)
    }

    pub fn get_versioned(&self, quotient: u64) -> Option<RwLockReadGuard<'_, VersionedItem<T>>> {
        self.quotients.get_versioned(quotient)
    }
}

impl<T: VersionedVecItem> TreeMapVecNode<T> {
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

    pub fn delete(&self, version: VersionNumber, quotient: u64, id: T::Id)
    where
        T::Id: Eq,
    {
        self.quotients.delete(version, quotient, id);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get(&self, quotient: u64) -> Option<RwLockReadGuard<'_, VersionedVec<T>>> {
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

impl<T: VersionedVecItem> Default for QuotientsMapVec<T> {
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
                quotient.value.write().insert(version, value);
            },
            |value| {
                Arc::new(Quotient {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    value: RwLock::new(VersionedItem::new(version, value)),
                })
            },
        );
    }

    pub fn delete(&self, version: VersionNumber, quotient: u64) {
        self.map.with_value(&quotient, |quotient| {
            quotient.value.write().delete(version);
        });
    }

    fn get_latest(&self, quotient: u64) -> Option<&T> {
        let op_val = self.map.lookup(&quotient).map(|q| {
            // SAFETY: here we are changing the lifetime of the value by using
            // `std::mem::transmute`, which by definition is not safe, but given our use of
            // this value and how we destroy it, its actually safe in this context.
            unsafe { std::mem::transmute::<Option<&T>, Option<&T>>(q.value.read().latest()) }
        });

        op_val.and_then(|op_val| op_val)
    }

    fn get_versioned(&self, quotient: u64) -> Option<RwLockReadGuard<'_, VersionedItem<T>>> {
        self.map.lookup(&quotient).map(|q| {
            // SAFETY: here we are changing the lifetime of the value by using
            // `std::mem::transmute`, which by definition is not safe, but given our use of
            // this value and how we destroy it, its actually safe in this context.
            unsafe { std::mem::transmute(q.value.read()) }
        })
    }

    fn latest_values(&self) -> Vec<&T> {
        self.map
            .to_list()
            .into_iter()
            .map(|(_, v)| {
                // SAFETY: here we are changing the lifetime of the value by using
                // `std::mem::transmute`, which by definition is not safe, but given our use of
                // this value and how we destroy it, its actually safe in this context.
                unsafe { std::mem::transmute::<Option<&T>, Option<&T>>(v.value.read().latest()) }
            })
            .take_while(|v| v.is_some())
            .map(|v| v.unwrap())
            .collect()
    }
}

impl<T: VersionedVecItem> QuotientsMapVec<T> {
    pub fn push(&self, version: VersionNumber, quotient: u64, value: T) {
        self.map.modify_or_insert_with_value(
            quotient,
            value,
            |value, quotient| {
                quotient.value.write().push(version, value);
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

    pub fn delete(&self, version: VersionNumber, quotient: u64, id: T::Id)
    where
        T::Id: Eq,
    {
        self.map.with_value(&quotient, |quotient| {
            quotient.value.write().delete(version, id);
        });
    }

    fn get(&self, quotient: u64) -> Option<RwLockReadGuard<'_, VersionedVec<T>>> {
        self.map.lookup(&quotient).map(|q| unsafe {
            std::mem::transmute::<
                RwLockReadGuard<'_, VersionedVec<T>>,
                RwLockReadGuard<'_, VersionedVec<T>>,
            >(q.value.read())
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
impl<K: TreeMapKey + Clone, V> PartialEq for TreeMapVec<K, V> {
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
impl<K: std::fmt::Debug + TreeMapKey + Clone + Ord, V> std::fmt::Debug for TreeMapVec<K, V>
where
    V: std::fmt::Debug + VersionedVecItem,
    <V as VersionedVecItem>::Id: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapVec")
            .field("root", &self.root)
            .finish()
    }
}

impl<K: TreeMapKey, V> TreeMap<K, V> {
    /// Constructor to create a new TreeMap
    ///
    /// @NOTE: In order to load an existing TreeMap that's already
    /// been serialized to disk, use `TreeMap::deserialize` method
    /// from the `SimpleSerialize` trait implementation.
    pub fn new(
        dim_bufman: BufferManager,
        data_bufmans: BufferManagerFactory<VersionNumber>,
    ) -> Self {
        Self {
            root: TreeMapNode::new(0),
            dim_bufman,
            data_bufmans,
            _marker: PhantomData,
        }
    }

    pub fn insert(&self, version: VersionNumber, key: &K, value: V) {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.insert(version, key, value);
    }

    pub fn delete(&self, version: VersionNumber, key: &K) {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.delete(version, key);
    }

    pub fn get_latest(&self, key: &K) -> Option<&V> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_latest(key)
    }

    pub fn get_versioned(&self, key: &K) -> Option<RwLockReadGuard<'_, VersionedItem<V>>> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_versioned(key)
    }

    /// Returns an iterator over all values in the TreeMap
    ///
    /// @NOTE: There's no equivalent iterator for keys because the
    /// TreeMap doesn't actually store keys in it's raw form (keys are
    /// hashed and stored). One way to workaround this is to store the
    /// key additionally in every value, so that iterating over the
    /// values also obtains the keys.
    pub fn latest_values(&self) -> TreeMapValuesIter<'_, V> {
        TreeMapValuesIter::new(&self.root)
    }
}

impl<K, V: SimpleSerialize> TreeMap<K, V> {
    pub fn serialize(&self) -> Result<(), BufIoError> {
        let cursor = self.dim_bufman.open_cursor()?;
        self.dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let offset = self
            .root
            .serialize(&self.dim_bufman, &self.data_bufmans, cursor)?;
        self.dim_bufman.seek_with_cursor(cursor, 0)?;
        self.dim_bufman.update_u32_with_cursor(cursor, offset)?;
        self.dim_bufman.close_cursor(cursor)?;
        self.dim_bufman.flush()?;
        self.data_bufmans.flush_all()?;
        Ok(())
    }

    pub fn deserialize(
        dim_bufman: BufferManager,
        data_bufmans: BufferManagerFactory<VersionNumber>,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        let offset = dim_bufman.read_u32_with_cursor(cursor)?;
        dim_bufman.close_cursor(cursor)?;
        Ok(Self {
            root: TreeMapNode::deserialize(
                &dim_bufman,
                &data_bufmans,
                FileOffset(offset),
                VersionNumber::from(u32::MAX), // not used
            )?,
            dim_bufman,
            data_bufmans,
            _marker: PhantomData,
        })
    }
}

pub struct TreeMapValuesIter<'a, T> {
    stack: VecDeque<&'a TreeMapNode<T>>,
    curr_iter: Option<(u16, std::vec::IntoIter<&'a T>)>,
}

impl<'a, T> TreeMapValuesIter<'a, T> {
    pub fn new(root_node: &'a TreeMapNode<T>) -> Self {
        let mut stack = VecDeque::new();
        stack.push_front(root_node);
        Self {
            stack,
            curr_iter: None,
        }
    }
}

impl<'a, T: std::fmt::Debug> Iterator for TreeMapValuesIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // @NOTE: Keeping node idx in state helps with debugging
            if let Some((_nidx, iter)) = &mut self.curr_iter {
                if let Some(item) = iter.next() {
                    return Some(item);
                }
            }

            if let Some(node) = self.stack.pop_front() {
                let node_values = node.quotients.latest_values().into_iter();
                self.curr_iter = Some((node.node_idx, node_values));
                for child_ptr in &node.children {
                    if !child_ptr.is_null() {
                        unsafe {
                            let child = &*child_ptr;
                            self.stack.push_front(child)
                        }
                    }
                }
            } else {
                return None;
            }
        }
    }
}

impl<K: TreeMapKey, V: VersionedVecItem> TreeMapVec<K, V> {
    pub fn new(
        dim_bufman: BufferManager,
        data_bufmans: BufferManagerFactory<VersionNumber>,
    ) -> Self {
        Self {
            root: TreeMapVecNode::new(0),
            dim_bufman,
            data_bufmans,
            _marker: PhantomData,
        }
    }

    pub fn push(&self, version: VersionNumber, key: &K, value: V) {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.push(version, key, value);
    }

    pub fn delete(&self, version: VersionNumber, key: &K, id: V::Id)
    where
        V::Id: Eq,
    {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.delete(version, key, id);
    }

    pub fn get(&self, key: &K) -> Option<RwLockReadGuard<'_, VersionedVec<V>>> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get(key)
    }
}

impl<K, V> TreeMapVec<K, V>
where
    V: SimpleSerialize + VersionedVecItem,
    <V as VersionedVecItem>::Id: SimpleSerialize,
{
    pub fn serialize(&self) -> Result<(), BufIoError> {
        let cursor = self.dim_bufman.open_cursor()?;
        self.dim_bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let offset = self
            .root
            .serialize(&self.dim_bufman, &self.data_bufmans, cursor)?;
        self.dim_bufman.seek_with_cursor(cursor, 0)?;
        self.dim_bufman.update_u32_with_cursor(cursor, offset)?;
        self.dim_bufman.close_cursor(cursor)?;
        self.dim_bufman.flush()?;
        self.data_bufmans.flush_all()?;
        Ok(())
    }

    pub fn deserialize(
        dim_bufman: BufferManager,
        data_bufmans: BufferManagerFactory<VersionNumber>,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        let offset = dim_bufman.read_u32_with_cursor(cursor)?;
        dim_bufman.close_cursor(cursor)?;
        Ok(Self {
            root: TreeMapVecNode::deserialize(
                &dim_bufman,
                &data_bufmans,
                FileOffset(offset),
                VersionNumber::from(u32::MAX), // not used
            )?,
            dim_bufman,
            data_bufmans,
            _marker: PhantomData,
        })
    }
}

impl TreeMapKey for u64 {
    fn key(&self) -> u64 {
        *self
    }
}

impl TreeMapKey for String {
    fn key(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs::OpenOptions};

    use rand::Rng;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn tests_basic_usage() {
        let tempdir = tempdir().unwrap();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(false)
            .create(true)
            .open(tempdir.as_ref().join("tree_map.dim"))
            .unwrap();
        let dim_bufman = BufferManager::new(file, 8192).unwrap();
        let data_bufmans = BufferManagerFactory::new(
            tempdir.as_ref().into(),
            |root, version: &VersionNumber| root.join(format!("tree_map.{}.data", **version)),
            8192,
        );
        let map: TreeMap<u64, u64> = TreeMap::new(dim_bufman, data_bufmans);
        map.insert(0.into(), &65536, 0);
        map.insert(0.into(), &0, 23);
        assert_eq!(map.get_latest(&0), Some(&23));
        map.insert(1.into(), &0, 29);
        assert_eq!(map.get_latest(&0), Some(&29));
        assert_eq!(
            map.get_versioned(&0).as_deref(),
            Some(&VersionedItem {
                version: 0.into(),
                serialized_at: RwLock::new(None),
                value: Some(23),
                next: Some(Box::new(VersionedItem::new(1.into(), 29)))
            })
        );
    }

    #[test]
    fn test_latest_values() {
        let tempdir = tempdir().unwrap();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(false)
            .create(true)
            .open(tempdir.as_ref().join("tree_map-2.dim"))
            .unwrap();
        let dim_bufman = BufferManager::new(file, 8192).unwrap();
        let data_bufmans = BufferManagerFactory::new(
            tempdir.as_ref().into(),
            |root, version: &VersionNumber| root.join(format!("tree_map-2.{}.data", **version)),
            8192,
        );
        let map: TreeMap<u64, u64> = TreeMap::new(dim_bufman, data_bufmans);

        let mut rng = rand::thread_rng();

        // Test with u16::MAX random values between 1 and u64::MAX
        let count = u16::MAX as usize;

        let test_values: Vec<u64> = (0..count).map(|_| rng.gen_range(1..=u64::MAX)).collect();

        for i in &test_values {
            map.insert(0.into(), i, *i);
        }

        let result: HashSet<&u64> = map.latest_values().collect();

        assert_eq!(count, result.len());

        for i in test_values {
            assert!(result.contains(&i));
        }
    }

    #[test]
    fn test_treemap_latest_values_same_version() {
        let tempdir = tempdir().unwrap();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(false)
            .create(true)
            .open(tempdir.as_ref().join("tree_map-3.dim"))
            .unwrap();
        let dim_bufman = BufferManager::new(file, 8192).unwrap();
        let data_bufmans = BufferManagerFactory::new(
            tempdir.as_ref().into(),
            |root, version: &VersionNumber| root.join(format!("tree_map-3.{}.data", **version)),
            8192,
        );
        let map: TreeMap<u64, u64> = TreeMap::new(dim_bufman, data_bufmans);

        map.insert(0.into(), &1, 100);
        map.insert(0.into(), &2, 200);

        map.serialize().unwrap();

        // overwrite without changing version
        map.insert(0.into(), &2, 250);

        map.serialize().unwrap();

        let result: HashSet<u64> = map.latest_values().copied().collect();
        assert!(result.contains(&100));
        assert!(result.contains(&250));
        assert!(!result.contains(&200));
    }

    #[test]
    fn test_treemap_latest_values_from_disk() {
        let tempdir = tempdir().unwrap();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(tempdir.as_ref().join("tree_map-4.dim"))
            .unwrap();
        let dim_bufman = BufferManager::new(file, 8192).unwrap();
        let data_bufmans = BufferManagerFactory::new(
            tempdir.as_ref().into(),
            |root, version: &VersionNumber| root.join(format!("tree_map-4.{}.data", **version)),
            8192,
        );
        let map: TreeMap<u64, u64> = TreeMap::new(dim_bufman, data_bufmans);
        map.insert(0.into(), &1, 100);
        map.insert(0.into(), &20, 2000);
        map.insert(0.into(), &300, 30000);
        map.insert(0.into(), &4000, 400000);

        map.serialize().unwrap();

        drop(map);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(false)
            .open(tempdir.as_ref().join("tree_map-4.dim"))
            .unwrap();
        let dim_bufman = BufferManager::new(file, 8192).unwrap();
        let data_bufmans = BufferManagerFactory::new(
            tempdir.as_ref().into(),
            |root, version: &VersionNumber| root.join(format!("tree_map-4.{}.data", **version)),
            8192,
        );
        let map: TreeMap<u64, u64> = TreeMap::deserialize(dim_bufman, data_bufmans).unwrap();

        let result: HashSet<u64> = map.latest_values().copied().collect();

        assert!(result.contains(&100));
        assert!(result.contains(&2000));
        assert!(result.contains(&30000));
        assert!(result.contains(&400000));
    }
}
