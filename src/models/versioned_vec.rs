use std::{marker::PhantomData, sync::RwLock};

use super::{
    buffered_io::{BufIoError, BufferManager},
    types::{FileOffset, InternalId},
    versioning::VersionNumber,
};

pub trait StorageTrait: Copy + Eq + std::fmt::Debug {
    const TOMBSTONE: Self;
    const DELETE_MARKER_FLAG: Self;
    const BYTES_SIZE: usize;

    fn is_tombstone(&self) -> bool;
    fn is_delete_marker(&self) -> bool;
    fn build_delete_marker(version: VersionNumber, index: u32) -> Self;
    fn get_delete_marker_fields(&self) -> (VersionNumber, u32);
    fn serialize(&self, vec: &mut Vec<u8>);
    fn deserialize(bufman: &BufferManager, cursor: u64) -> Result<Self, BufIoError>;
}

impl StorageTrait for u64 {
    const TOMBSTONE: Self = u64::MAX;
    const DELETE_MARKER_FLAG: Self = 1 << 63;
    const BYTES_SIZE: usize = 8;

    fn is_tombstone(&self) -> bool {
        *self == Self::TOMBSTONE
    }

    fn is_delete_marker(&self) -> bool {
        (*self & Self::DELETE_MARKER_FLAG) != 0 && !self.is_tombstone()
    }

    fn build_delete_marker(version: VersionNumber, index: u32) -> Self {
        Self::DELETE_MARKER_FLAG | ((*version as u64) << 32) | (index as u64)
    }

    fn get_delete_marker_fields(&self) -> (VersionNumber, u32) {
        let without_flag = *self & !Self::DELETE_MARKER_FLAG;
        let version = VersionNumber::from((without_flag >> 32) as u32);
        let index = without_flag as u32;
        (version, index)
    }

    fn serialize(&self, vec: &mut Vec<u8>) {
        vec.extend(self.to_le_bytes());
    }

    fn deserialize(bufman: &BufferManager, cursor: u64) -> Result<Self, BufIoError> {
        bufman.read_u64_with_cursor(cursor)
    }
}

impl StorageTrait for u128 {
    const TOMBSTONE: Self = u128::MAX;
    const DELETE_MARKER_FLAG: Self = 1 << 127;
    const BYTES_SIZE: usize = 16;

    fn is_tombstone(&self) -> bool {
        *self == Self::TOMBSTONE
    }

    fn is_delete_marker(&self) -> bool {
        (*self & Self::DELETE_MARKER_FLAG) != 0 && !self.is_tombstone()
    }

    fn build_delete_marker(version: VersionNumber, index: u32) -> Self {
        Self::DELETE_MARKER_FLAG | ((*version as u128) << 32) | (index as u128)
    }

    fn get_delete_marker_fields(&self) -> (VersionNumber, u32) {
        let without_flag = *self & !Self::DELETE_MARKER_FLAG;
        let version = VersionNumber::from((without_flag >> 32) as u32);
        let index = without_flag as u32;
        (version, index)
    }

    fn serialize(&self, vec: &mut Vec<u8>) {
        vec.extend(self.to_le_bytes());
    }

    fn deserialize(bufman: &BufferManager, cursor: u64) -> Result<Self, BufIoError> {
        bufman.read_u128_with_cursor(cursor)
    }
}

pub trait VersionedVecItem {
    type Storage: StorageTrait;
    type Id;

    fn id(storage: Self::Storage) -> Self::Id;
    fn into_storage(self) -> Self::Storage;
    fn from_storage(storage: Self::Storage) -> Self;
}

// Existing implementations using u64 storage
impl VersionedVecItem for (u32, f32) {
    type Storage = u64;
    type Id = u32;

    fn id(storage: u64) -> Self::Id {
        (storage >> 32) as u32
    }

    fn into_storage(self) -> u64 {
        ((self.0 as u64) << 32) | (self.1.to_bits() as u64)
    }

    fn from_storage(storage: u64) -> Self {
        ((storage >> 32) as u32, f32::from_bits(storage as u32))
    }
}

impl VersionedVecItem for u32 {
    type Storage = u64;
    type Id = u32;

    fn id(storage: u64) -> Self::Id {
        storage as u32
    }

    fn into_storage(self) -> u64 {
        self as u64
    }

    fn from_storage(storage: u64) -> Self {
        storage as u32
    }
}

impl VersionedVecItem for u16 {
    type Storage = u64;
    type Id = u16;

    fn id(storage: u64) -> Self::Id {
        storage as u16
    }

    fn into_storage(self) -> u64 {
        self as u64
    }

    fn from_storage(storage: u64) -> Self {
        storage as u16
    }
}

impl VersionedVecItem for InternalId {
    type Storage = u64;
    type Id = InternalId;

    fn id(storage: u64) -> Self::Id {
        Self::from(storage as u32)
    }

    fn into_storage(self) -> u64 {
        *self as u64
    }

    fn from_storage(storage: u64) -> Self {
        Self::from(storage as u32)
    }
}

pub struct VersionedVec<T: VersionedVecItem> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: VersionNumber,
    pub list: Vec<T::Storage>,
    pub next: Option<Box<VersionedVec<T>>>,
    pub _marker: PhantomData<T>,
}

unsafe impl<T: Send + VersionedVecItem> Send for VersionedVec<T> {}
unsafe impl<T: Sync + VersionedVecItem> Sync for VersionedVec<T> {}

#[cfg(test)]
impl<T: VersionedVecItem> Clone for VersionedVec<T>
where
    T::Storage: Clone,
{
    fn clone(&self) -> Self {
        Self {
            serialized_at: RwLock::new(*self.serialized_at.read().unwrap()),
            version: self.version,
            list: self.list.clone(),
            next: self.next.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: VersionedVecItem> VersionedVec<T> {
    pub fn new(version: VersionNumber) -> Self {
        Self {
            serialized_at: RwLock::new(None),
            version,
            list: Vec::new(),
            next: None,
            _marker: PhantomData,
        }
    }

    pub fn push(&mut self, version: VersionNumber, value: T) {
        if self.version == version {
            return self.list.push(value.into_storage());
        }

        if let Some(next) = &mut self.next {
            next.push(version, value);
        } else {
            let mut new_next = Box::new(Self::new(version));
            new_next.push(version, value);
            self.next = Some(new_next);
        }
    }

    pub fn delete(&mut self, version: VersionNumber, id: T::Id)
    where
        T::Id: Eq,
    {
        let Some((inserted_version, idx)) = self.find_and_mark_deleted(id) else {
            return;
        };

        self.delete_internal(version, inserted_version, idx);
    }

    fn find_and_mark_deleted(&mut self, id: T::Id) -> Option<(VersionNumber, u32)>
    where
        T::Id: Eq,
    {
        let idx = self.list.iter().position(|item| T::id(*item) == id);
        if let Some(idx) = idx {
            self.list[idx] = T::Storage::TOMBSTONE;
            return Some((self.version, idx as u32));
        }
        self.next
            .as_mut()
            .and_then(|next| next.find_and_mark_deleted(id))
    }

    fn delete_internal(
        &mut self,
        version: VersionNumber,
        inserted_version: VersionNumber,
        inserted_idx: u32,
    ) {
        if self.version == version {
            if inserted_version == self.version {
                self.list.remove(inserted_idx as usize);
            } else {
                self.list.push(T::Storage::build_delete_marker(
                    inserted_version,
                    inserted_idx,
                ));
            }
            return;
        }

        if let Some(next) = &mut self.next {
            next.delete_internal(version, inserted_version, inserted_idx);
        } else {
            let mut new_next = Box::new(Self::new(version));
            new_next.delete_internal(version, inserted_version, inserted_idx);
            self.next = Some(new_next);
        }
    }

    pub fn iter(&self) -> VersionedVecIter<'_, T> {
        let iter = self.list.iter();
        VersionedVecIter {
            current_version: self.version,
            current_iter: iter,
            next: self.next.as_deref(),
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        let current_len = self.list.len();

        match &self.next {
            Some(next_node) => current_len + next_node.len(),
            None => current_len,
        }
    }

    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl VersionedVec<(u32, f32)> {
    pub fn push_sorted(&mut self, version: VersionNumber, value: (u32, f32)) {
        if self.version == version {
            let storage_value = value.into_storage();
            let mut i = self.list.len();
            while i > 0 && <(u32, f32)>::id(self.list[i - 1]) > value.0 {
                i -= 1;
            }
            self.list.insert(i, storage_value);
            return;
        }

        if let Some(next) = &mut self.next {
            next.push_sorted(version, value);
        } else {
            let mut new_next = Box::new(Self::new(version));
            new_next.push_sorted(version, value);
            self.next = Some(new_next);
        }
    }
}

impl<T: VersionedVecItem> PartialEq for VersionedVec<T>
where
    T::Storage: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version && self.list == other.list && self.next == other.next
    }
}

impl<T: VersionedVecItem> std::fmt::Debug for VersionedVec<T>
where
    T::Storage: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VersionedVec")
            .field("version", &self.version)
            .field("list", &self.list)
            .field("next", &self.next)
            .finish()
    }
}

pub struct VersionedVecIter<'a, T: VersionedVecItem> {
    current_iter: std::slice::Iter<'a, T::Storage>,
    current_version: VersionNumber,
    next: Option<&'a VersionedVec<T>>,
    _marker: PhantomData<T>,
}

impl<T: VersionedVecItem> Iterator for VersionedVecIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        for storage in self.current_iter.by_ref() {
            if storage.is_tombstone() || storage.is_delete_marker() {
                continue;
            }
            return Some(T::from_storage(*storage));
        }

        if let Some(next_node) = self.next {
            self.current_version = next_node.version;
            self.current_iter = next_node.list.iter();
            self.next = next_node.next.as_deref();
            self.next()
        } else {
            None
        }
    }
}
