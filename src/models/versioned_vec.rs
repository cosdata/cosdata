use std::{hash::Hash, sync::RwLock};

use rustc_hash::FxHashMap;

use super::{
    types::{FileOffset, InternalId},
    versioning::VersionNumber,
};

pub trait VersionedVecItem {
    type Id;

    fn id(&self) -> &Self::Id;
}

impl VersionedVecItem for (u32, f32) {
    type Id = u32;

    fn id(&self) -> &Self::Id {
        &self.0
    }
}

impl VersionedVecItem for u32 {
    type Id = u32;

    fn id(&self) -> &Self::Id {
        self
    }
}

impl VersionedVecItem for u16 {
    type Id = u16;

    fn id(&self) -> &Self::Id {
        self
    }
}

impl VersionedVecItem for InternalId {
    type Id = InternalId;

    fn id(&self) -> &Self::Id {
        self
    }
}

pub struct VersionedVec<T: VersionedVecItem> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: VersionNumber,
    pub list: Vec<T>,
    pub deleted: Vec<T::Id>,
    pub next: Option<Box<VersionedVec<T>>>,
}

unsafe impl<T: Send + VersionedVecItem> Send for VersionedVec<T> {}
unsafe impl<T: Sync + VersionedVecItem> Sync for VersionedVec<T> {}

#[cfg(test)]
impl<T> Clone for VersionedVec<T>
where
    T: Clone + VersionedVecItem,
    <T as VersionedVecItem>::Id: Clone,
{
    fn clone(&self) -> Self {
        Self {
            serialized_at: RwLock::new(*self.serialized_at.read().unwrap()),
            version: self.version,
            list: self.list.clone(),
            deleted: self.deleted.clone(),
            next: self.next.clone(),
        }
    }
}

impl<T: VersionedVecItem> VersionedVec<T> {
    pub fn new(version: VersionNumber) -> VersionedVec<T> {
        Self {
            serialized_at: RwLock::new(None),
            version,
            list: Vec::new(),
            deleted: Vec::new(),
            next: None,
        }
    }

    pub fn push(&mut self, version: VersionNumber, value: T) {
        if self.version == version {
            return self.list.push(value);
        }

        if let Some(next) = &mut self.next {
            next.push(version, value);
        } else {
            let mut new_next = Box::new(Self::new(version));
            new_next.push(version, value);
            self.next = Some(new_next);
        }
    }

    pub fn delete(&mut self, version: VersionNumber, value: T::Id) {
        if self.version == version {
            return self.deleted.push(value);
        }

        if let Some(next) = &mut self.next {
            next.delete(version, value);
        } else {
            let mut new_next = Box::new(Self::new(version));
            new_next.delete(version, value);
            self.next = Some(new_next);
        }
    }

    pub fn iter(&self) -> VersionedVecIter<'_, T>
    where
        <T as VersionedVecItem>::Id: Hash + Eq + Copy,
    {
        let iter = self.list.iter();
        let mut deleted = FxHashMap::default();
        self.collect_deleted_items(&mut deleted);
        VersionedVecIter {
            current_version: self.version,
            current_iter: iter,
            deleted,
            next: self.next.as_deref(),
        }
    }

    fn collect_deleted_items(&self, set: &mut FxHashMap<<T as VersionedVecItem>::Id, VersionNumber>)
    where
        <T as VersionedVecItem>::Id: Hash + Eq + Copy,
    {
        for deleted in &self.deleted {
            set.insert(*deleted, self.version);
        }
        if let Some(next) = &self.next {
            next.collect_deleted_items(set);
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
            while i > 0 && self.list[i - 1].0 > value.0 {
                i -= 1;
            }
            self.list.insert(i, value);
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

impl<T> PartialEq for VersionedVec<T>
where
    T: PartialEq + VersionedVecItem,
    <T as VersionedVecItem>::Id: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.list == other.list
            && self.deleted == other.deleted
            && self.next == other.next
    }
}

impl<T> std::fmt::Debug for VersionedVec<T>
where
    T: std::fmt::Debug + VersionedVecItem,
    <T as VersionedVecItem>::Id: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeVersionedList")
            .field("version", &self.version)
            .field("list", &self.list)
            .field("next", &self.next)
            .finish()
    }
}

// Iterator over &T
pub struct VersionedVecIter<'a, T: VersionedVecItem> {
    current_iter: std::slice::Iter<'a, T>,
    current_version: VersionNumber,
    deleted: FxHashMap<T::Id, VersionNumber>,
    next: Option<&'a VersionedVec<T>>,
}

impl<'a, T: VersionedVecItem> VersionedVecIter<'a, T> {
    #[inline(always)]
    fn next_with_skips(&mut self) -> Option<&'a T> {
        if let Some(item) = self.current_iter.next() {
            return Some(item);
        }

        if let Some(next_node) = self.next {
            self.current_version = next_node.version;
            self.current_iter = next_node.list.iter();
            self.next = next_node.next.as_deref();
            self.current_iter.next()
        } else {
            None
        }
    }
}

impl<'a, T> Iterator for VersionedVecIter<'a, T>
where
    T: VersionedVecItem,
    <T as VersionedVecItem>::Id: Hash + Eq,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next = self.next_with_skips()?;
        if self.deleted.is_empty() {
            return Some(next);
        }

        while let Some(version) = self.deleted.get(next.id()) {
            if *self.current_version > **version {
                break;
            }
            next = self.next_with_skips()?;
        }

        Some(next)
    }
}
