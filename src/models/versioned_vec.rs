use std::{marker::PhantomData, sync::RwLock};

use super::{
    types::{FileOffset, InternalId},
    versioning::VersionNumber,
};

pub trait VersionedVecItem {
    type Id;

    fn id(storage: u64) -> Self::Id;

    fn into_storage(self) -> u64;

    fn from_storage(storage: u64) -> Self;
}

impl VersionedVecItem for (u32, f32) {
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

pub struct VersionedVec<T> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: VersionNumber,
    pub list: Vec<u64>,
    pub next: Option<Box<VersionedVec<T>>>,
    pub _marker: PhantomData<T>,
}

unsafe impl<T: Send + VersionedVecItem> Send for VersionedVec<T> {}
unsafe impl<T: Sync + VersionedVecItem> Sync for VersionedVec<T> {}

#[cfg(test)]
impl<T> Clone for VersionedVec<T> {
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
    pub fn new(version: VersionNumber) -> VersionedVec<T> {
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
        <T as VersionedVecItem>::Id: Eq,
    {
        let Some((inserted_version, idx)) = self.find_and_mark_deleted(id) else {
            return;
        };

        self.delete_internal(version, inserted_version, idx);
    }

    fn find_and_mark_deleted(&mut self, id: T::Id) -> Option<(VersionNumber, u32)>
    where
        <T as VersionedVecItem>::Id: Eq,
    {
        let idx = self.list.iter().position(|item| T::id(*item) == id);
        if let Some(idx) = idx {
            self.list[idx] = u64::MAX;
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
    ) where
        <T as VersionedVecItem>::Id: Eq,
    {
        if self.version == version {
            if inserted_version == self.version {
                self.list.remove(inserted_idx as usize);
            } else {
                self.list
                    .push((1 << 63) | ((*inserted_version as u64) << 32) | (inserted_idx as u64))
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
            let mut i = self.list.len();
            while i > 0 && <(u32, f32)>::from_storage(self.list[i - 1]).0 > value.0 {
                i -= 1;
            }
            self.list.insert(i, value.into_storage());
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

impl<T> PartialEq for VersionedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version && self.list == other.list && self.next == other.next
    }
}

impl<T> std::fmt::Debug for VersionedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeVersionedList")
            .field("version", &self.version)
            .field("list", &self.list)
            .field("next", &self.next)
            .finish()
    }
}

// Iterator over &T
pub struct VersionedVecIter<'a, T> {
    current_iter: std::slice::Iter<'a, u64>,
    current_version: VersionNumber,
    next: Option<&'a VersionedVec<T>>,
}

impl<T: VersionedVecItem> Iterator for VersionedVecIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        for item in self.current_iter.by_ref() {
            let storage = *item;

            // Skip tombstones (MSB set) and u64::MAX
            if storage == u64::MAX || (storage & (1 << 63)) != 0 {
                continue;
            }

            return Some(T::from_storage(storage));
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
