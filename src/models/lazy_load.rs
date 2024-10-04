use super::cache_loader::NodeRegistry;
use super::common::WaCustomError;
use super::identity_collections::{Identifiable, IdentityMap, IdentityMapKey, IdentitySet};
use super::serializer::CustomSerialize;
use super::types::{FileOffset, STM};
use super::versioning::*;
use arcshift::ArcShift;
use core::panic;
use std::fmt;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn largest_power_of_4_below(x: u32) -> u32 {
    // This function is used to calculate the largest power of 4 (4^n) such that
    // 4^n <= x, where x represents the gap between the current version and the
    // target version in our version control system.
    //
    // The system uses an exponentially spaced versioning scheme, where each
    // checkpoint is spaced by powers of 4 (1, 4, 16, 64, etc.). This minimizes
    // the number of intermediate versions stored, allowing efficient lookups
    // and updates by focusing only on meaningful checkpoints.
    //
    // The input x should not be zero because finding a "largest power of 4 below zero"
    // is undefined, as zero does not have any significant bits for such a calculation.
    assert_ne!(x, 0, "x should not be zero");

    let msb_position = 31 - x.leading_zeros(); // Find the most significant bit's position
    msb_position / 2 // Return the power index of the largest 4^n ≤ x
}

pub trait SyncPersist {
    fn set_persistence(&self, flag: bool);
    fn needs_persistence(&self) -> bool;
    fn get_current_version(&self) -> Hash;
}

pub const CHUNK_SIZE: usize = 5;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum FileIndex {
    Valid { offset: FileOffset, version: Hash },
    Invalid,
}

#[derive(Clone)]
// As the name suggests, this is a wrapper for lazy-loading the inner data. Its
// serialization/deserialization mechanisms are designed to handle cyclic data while minimizing
// redundant re-serializations, ensuring optimal performance.
pub enum LazyItem<T: Clone + 'static> {
    Valid {
        // Holds the actual data. Wrapped in an `Option` to indicate whether the data is loaded.
        data: Option<ArcShift<T>>,
        // Pointer to the file offset where the data is stored. Used for lazy loading the data when
        // needed. If the data is not loaded, this index retrieves it from persistent storage.
        file_index: ArcShift<Option<FileIndex>>,
        // Tracks the lifespan the item.
        decay_counter: usize,
        // Prevents infinite serialization loops when handling cyclic data. This flag indicates
        // whether the `LazyItem` has already been serialized during the current cycle, ensuring
        // the data isn't serialized multiple times. It must be reset after the whole serialization
        // cycle is complete.
        persist_flag: Arc<AtomicBool>,
        // Each element in `versions` is `4^n` versions ahead of the current version:
        // - 0th element is 1 version ahead (`4^0`)
        // - 1st element is 4 versions ahead (`4^1`)
        // - 2nd element is 16 versions ahead (`4^2`), etc.
        //
        // This exponential spacing aids in:
        // 1. **Efficient lookups**: The predictable gaps make version navigation fast.
        // 2. **Scalability**: The exponential structure keeps the history compact for large
        // version sets.
        //
        // `LazyItemVec` is ideal for maintaining sequential versions with fast access by index.
        versions: LazyItemVec<T>,
        // Hash value representing the current version.
        version_id: Hash,
        // Indicates if this `LazyItem` has ever been serialized. Unlike `persist_flag`, this flag
        // should not be reset. It avoids re-serializing the entire `LazyItem` if it's already
        // been serialized once, as only the `versions` field might change.
        serialized_flag: Arc<AtomicBool>,
    },
    // Marks the item as invalid or unavailable.
    Invalid,
}

#[derive(Clone)]
pub struct EagerLazyItem<T: Clone + 'static, E: Clone + CustomSerialize + 'static>(
    pub E,
    pub LazyItem<T>,
);

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum LazyItemId {
    Memory(u64),
    Persist(FileIndex),
}

impl fmt::Display for FileIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileIndex::Valid { offset, version } => {
                write!(f, "FileIndex(offset: {}, version: {})", offset.0, **version)
            }
            FileIndex::Invalid => write!(f, "FileIndex(Invalid)"),
        }
    }
}

impl<T> Identifiable for LazyItem<T>
where
    T: Clone + Identifiable<Id = u64> + 'static,
{
    type Id = LazyItemId;

    fn get_id(&self) -> Self::Id {
        if let LazyItem::Valid {
            data, file_index, ..
        } = self
        {
            if let Some(offset) = file_index.clone().get().clone() {
                return LazyItemId::Persist(offset);
            }

            if let Some(data) = data {
                let mut arc = data.clone();
                return LazyItemId::Memory(arc.get().get_id());
            }
        }

        LazyItemId::Persist(FileIndex::Invalid)
    }
}

impl<T, E> Identifiable for EagerLazyItem<T, E>
where
    T: Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    type Id = LazyItemId;

    fn get_id(&self) -> Self::Id {
        self.1.get_id()
    }
}

#[derive(Clone)]
pub struct LazyItemRef<T>
where
    T: Clone + 'static,
{
    pub item: ArcShift<LazyItem<T>>,
}

#[derive(Clone)]
pub struct EagerLazyItemSet<T, E>
where
    T: Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    pub items: STM<IdentitySet<EagerLazyItem<T, E>>>,
}

#[derive(Clone)]
pub struct LazyItemSet<T>
where
    T: Clone + Identifiable<Id = u64> + 'static,
{
    pub items: STM<IdentitySet<LazyItem<T>>>,
}

#[derive(Clone)]
pub struct LazyItemMap<T: Clone + 'static> {
    // A map-like structure that uses `IdentityMap` to store `LazyItem`s.
    // This allows efficient access and management of items using unique keys.
    // `STM` implies concurrent updates and access management with Software Transactional Memory.
    //
    // Use Case:
    // - Suitable for scenarios requiring fast, direct access to items based on unique keys.
    // - Ideal for non-sequential or sparse data where items are frequently added or removed.
    // - Provides O(1) average time complexity for lookups, insertions, and deletions.
    pub items: STM<IdentityMap<LazyItem<T>>>,
}

#[derive(Clone)]
pub struct LazyItemVec<T: Clone + 'static> {
    // A vector-based structure to store `LazyItem`s in a sequential manner.
    // This allows for ordered access and is useful for data where indices follow a pattern,
    // such as powers of 4 in version control systems.
    // `STM` implies concurrent updates and access management with Software Transactional Memory.
    //
    // Use Case:
    // - Ideal for ordered or sequential data where the index has a meaningful relationship.
    // - Suitable for systems where data is accessed or updated in a linear, ordered fashion.
    // - Provides O(1) time complexity for access by index and efficient memory usage if managed properly.
    pub items: STM<Vec<LazyItem<T>>>,
}

#[derive(Clone)]
pub struct LazyItemArray<T: Clone + 'static, const N: usize> {
    // An array-based structure to store `LazyItem`s with a fixed size.
    // This allows for a fixed number of items to be stored, providing a predictable and
    // memory-efficient way to manage a specific number of items.
    // `STM` implies concurrent updates and access management with Software Transactional Memory.
    //
    // Use Case:
    // - Ideal for scenarios where a fixed number of items need to be managed, such as children in
    // an inverted index or versions in a version control system.
    // - Provides O(1) time complexity for access by index and efficient memory usage.
    pub items: STM<[Option<LazyItem<T>>; N]>,
}

impl<T: Clone + 'static> SyncPersist for LazyItem<T> {
    fn set_persistence(&self, flag: bool) {
        if let Self::Valid { persist_flag, .. } = self {
            persist_flag.store(flag, Ordering::SeqCst);
        }
    }

    fn needs_persistence(&self) -> bool {
        if let Self::Valid { persist_flag, .. } = self {
            persist_flag.load(Ordering::SeqCst)
        } else {
            false
        }
    }

    fn get_current_version(&self) -> Hash {
        if let Self::Valid { version_id, .. } = self {
            *version_id
        } else {
            0.into()
        }
    }
}

impl<T: Clone + 'static> LazyItem<T> {
    pub fn new(version_id: Hash, item: T) -> Self {
        Self::Valid {
            data: Some(ArcShift::new(item)),
            file_index: ArcShift::new(None),
            decay_counter: 0,
            persist_flag: Arc::new(AtomicBool::new(true)),
            versions: LazyItemVec::new(),
            version_id,
            serialized_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn new_invalid() -> Self {
        Self::Invalid
    }

    pub fn from_data(version_id: Hash, data: T) -> Self {
        LazyItem::Valid {
            data: Some(ArcShift::new(data)),
            file_index: ArcShift::new(None),
            decay_counter: 0,
            persist_flag: Arc::new(AtomicBool::new(true)),
            versions: LazyItemVec::new(),
            version_id,
            serialized_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn from_arcshift(version_id: Hash, item: ArcShift<T>) -> Self {
        Self::Valid {
            data: Some(item),
            file_index: ArcShift::new(None),
            decay_counter: 0,
            persist_flag: Arc::new(AtomicBool::new(true)),
            versions: LazyItemVec::new(),
            version_id,
            serialized_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid)
    }

    pub fn get_lazy_data(&self) -> Option<ArcShift<T>> {
        if let Self::Valid { data, .. } = self {
            return data.clone();
        }
        None
    }

    pub fn set_data(&mut self, new_data: T) {
        if let Self::Valid { data, .. } = self {
            *data = Some(ArcShift::new(new_data))
        }
    }

    pub fn get_file_index(&self) -> Option<FileIndex> {
        if let Self::Valid { file_index, .. } = self {
            return file_index.clone().get().clone();
        }
        None
    }

    pub fn set_file_index(&self, new_file_index: Option<FileIndex>) {
        if let Self::Valid { file_index, .. } = self {
            file_index.clone().update(new_file_index);
        }
    }

    pub fn add_version(
        &self,
        vcs: Arc<VersionControl>,
        version: u32,
        lazy_item: LazyItem<T>,
    ) -> lmdb::Result<()> {
        if let Self::Valid {
            versions,
            version_id,
            ..
        } = self
        {
            // Retrieve current version from LMDB
            let version_hash = vcs
                .get_version_hash(version_id)?
                .ok_or(lmdb::Error::NotFound)?;
            // Calculate the difference between the target version and the current version
            let target_diff = version - *version_hash.version;

            // Use the largest power of 4 below the target difference to find the appropriate
            // checkpoint for storing this new version.
            let index = largest_power_of_4_below(target_diff);

            // If a version already exists at the calculated index, recursively add the new version.
            if let Some(existing_version) = versions.get(index as usize) {
                return existing_version.add_version(vcs, version, lazy_item);
            } else {
                // Insert the new version at the calculated index if none exists there yet.
                versions.insert(index as usize, lazy_item);
            }
        }

        Ok(())
    }

    pub fn get_versions(&self) -> Option<LazyItemVec<T>> {
        if let Self::Valid { versions, .. } = self {
            Some(versions.clone())
        } else {
            None
        }
    }

    pub fn get_version(
        &self,
        vcs: Arc<VersionControl>,
        version: u32,
    ) -> lmdb::Result<Option<LazyItem<T>>> {
        if let Self::Valid {
            versions,
            version_id,
            ..
        } = self
        {
            let version_hash = vcs
                .get_version_hash(version_id)?
                .ok_or(lmdb::Error::NotFound)?;
            if version == *version_hash.version {
                return Ok(Some(self.clone()));
            }
            let target_diff = version - *version_hash.version;
            let index = largest_power_of_4_below(target_diff);
            let Some(version_at_index) = versions.get(index as usize) else {
                return Ok(None);
            };
            let power = 1u32 << (index * 2);
            if power == target_diff {
                Ok(Some(version_at_index))
            } else {
                version_at_index.get_version(vcs, version)
            }
        } else {
            Ok(None)
        }
    }

    pub fn get_latest_version(&self) -> LazyItem<T> {
        if let Self::Valid { versions, .. } = self {
            if let Some(last) = versions.last() {
                return last.get_latest_version();
            }
        }
        self.clone()
    }

    pub fn set_versions_persistence(&self, flag: bool) {
        self.set_persistence(flag);
        if let Some(versions) = self.get_versions() {
            let mut items_arc = versions.items.clone();
            for version in items_arc.get().iter() {
                version.set_versions_persistence(flag);
            }
        }
    }
}

impl<T: Clone + CustomSerialize + 'static> LazyItem<T> {
    pub fn try_get_data(&self, cache: Arc<NodeRegistry>) -> Result<ArcShift<T>, WaCustomError> {
        if let Self::Valid {
            data, file_index, ..
        } = self
        {
            if let Some(data) = data {
                return Ok(data.clone());
            }

            let Some(file_index) = file_index.clone().get().clone() else {
                return Err(WaCustomError::LazyLoadingError(
                    "LazyItem offset is None".to_string(),
                ));
            };

            let deserialized = cache.load_item(file_index)?;

            Ok(ArcShift::new(deserialized))
        } else {
            return Err(WaCustomError::LazyLoadingError(
                "LazyItem is invalid".to_string(),
            ));
        }
    }

    pub fn get_data(&self, cache: Arc<NodeRegistry>) -> ArcShift<T> {
        if let Self::Valid {
            data, file_index, ..
        } = self
        {
            if let Some(data) = data {
                return data.clone();
            }

            let file_index = file_index
                .clone()
                .get()
                .clone()
                .expect("LazyItem offset is None");

            let deserialized = cache.load_item(file_index).expect("Deserialization error");

            ArcShift::new(deserialized)
        } else {
            panic!("LazyItem is invalid");
        }
    }
}

impl<T: Clone + 'static> LazyItemRef<T> {
    pub fn new(version_id: Hash, item: T) -> Self {
        Self {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(ArcShift::new(item)),
                file_index: ArcShift::new(None),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions: LazyItemVec::new(),
                version_id,
                serialized_flag: Arc::new(AtomicBool::new(false)),
            }),
        }
    }

    pub fn new_invalid() -> Self {
        Self {
            item: ArcShift::new(LazyItem::Invalid),
        }
    }

    pub fn from_arcshift(version_id: Hash, item: ArcShift<T>) -> Self {
        Self {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(item),
                file_index: ArcShift::new(None),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions: LazyItemVec::new(),
                version_id,
                serialized_flag: Arc::new(AtomicBool::new(false)),
            }),
        }
    }

    pub fn from_lazy(item: LazyItem<T>) -> Self {
        Self {
            item: ArcShift::new(item),
        }
    }

    pub fn is_valid(&self) -> bool {
        let mut arc = self.item.clone();
        arc.get().is_valid()
    }

    pub fn is_invalid(&self) -> bool {
        let mut arc = self.item.clone();
        arc.get().is_invalid()
    }

    pub fn get_data(&self) -> Option<ArcShift<T>> {
        let mut arc = self.item.clone();
        if let LazyItem::Valid { data, .. } = arc.get() {
            return data.clone();
        }
        None
    }

    pub fn set_data(&self, new_data: T) {
        let mut arc = self.item.clone();

        arc.rcu(|item| {
            let (file_index, decay_counter, persist_flag, version_id, versions, serialized_flag) =
                if let LazyItem::Valid {
                    file_index,
                    decay_counter,
                    persist_flag,
                    version_id,
                    versions,
                    serialized_flag,
                    ..
                } = item
                {
                    (
                        file_index.clone(),
                        *decay_counter,
                        persist_flag.clone(),
                        *version_id,
                        versions.clone(),
                        serialized_flag.clone(),
                    )
                } else {
                    (
                        ArcShift::new(None),
                        0,
                        Arc::new(AtomicBool::new(true)),
                        0.into(),
                        LazyItemVec::new(),
                        Arc::new(AtomicBool::new(false)),
                    )
                };

            LazyItem::Valid {
                data: Some(ArcShift::new(new_data)),
                file_index,
                decay_counter,
                persist_flag,
                versions,
                version_id,
                serialized_flag,
            }
        });
    }

    pub fn set_file_index(&self, new_offset: Option<FileIndex>) {
        let mut arc = self.item.clone();

        arc.rcu(|item| {
            let (data, decay_counter, persist_flag, version_id, versions, serialized_flag) =
                if let LazyItem::Valid {
                    data,
                    decay_counter,
                    persist_flag,
                    version_id,
                    versions,
                    serialized_flag,
                    ..
                } = item
                {
                    (
                        data.clone(),
                        *decay_counter,
                        persist_flag.clone(),
                        *version_id,
                        versions.clone(),
                        serialized_flag.clone(),
                    )
                } else {
                    (
                        None,
                        0,
                        Arc::new(AtomicBool::new(true)),
                        0.into(),
                        LazyItemVec::new(),
                        Arc::new(AtomicBool::new(false)),
                    )
                };

            LazyItem::Valid {
                data,
                file_index: ArcShift::new(new_offset),
                decay_counter,
                persist_flag,
                versions,
                version_id,
                serialized_flag,
            }
        });
    }

    pub fn get_current_version(&self) -> Hash {
        let mut arc = self.item.clone();
        arc.get().get_current_version()
    }
}

impl<T, E> EagerLazyItemSet<T, E>
where
    T: Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    pub fn new() -> Self {
        Self {
            items: STM::new(IdentitySet::new(), 5, false),
        }
    }

    pub fn from_set(set: IdentitySet<EagerLazyItem<T, E>>) -> Self {
        Self {
            items: STM::new(set, 5, false),
        }
    }

    pub fn insert(&self, item: EagerLazyItem<T, E>) {
        let mut arc = self.items.clone();

        arc.transactional_update(|set| {
            let mut set = set.clone();
            set.insert(item.clone());
            set
        })
        .unwrap();
    }

    pub fn iter(&self) -> impl Iterator<Item = EagerLazyItem<T, E>> {
        let mut arc = self.items.clone();
        let vec: Vec<_> = arc.get().iter().map(Clone::clone).collect();
        vec.into_iter()
    }

    pub fn is_empty(&self) -> bool {
        let mut arc = self.items.clone();
        arc.get().is_empty()
    }

    pub fn len(&self) -> usize {
        let mut arc = self.items.clone();
        arc.get().len()
    }
}

impl<T: Clone + Identifiable<Id = u64> + 'static> LazyItemSet<T> {
    pub fn new() -> Self {
        Self {
            items: STM::new(IdentitySet::new(), 1, true),
        }
    }

    pub fn from_set(set: IdentitySet<LazyItem<T>>) -> Self {
        Self {
            items: STM::new(set, 1, true),
        }
    }

    pub fn insert(&self, item: LazyItem<T>) {
        let mut arc = self.items.clone();

        arc.transactional_update(|set| {
            let mut set = set.clone();
            set.insert(item.clone());
            set
        })
        .unwrap();
    }

    pub fn iter(&self) -> impl Iterator<Item = LazyItem<T>> {
        let mut arc = self.items.clone();
        let vec: Vec<_> = arc.get().iter().map(Clone::clone).collect();
        vec.into_iter()
    }

    pub fn is_empty(&self) -> bool {
        let mut arc = self.items.clone();
        arc.get().is_empty()
    }

    pub fn len(&self) -> usize {
        let mut arc = self.items.clone();
        arc.get().len()
    }
}

impl<T: Clone + 'static> LazyItemMap<T> {
    pub fn new() -> Self {
        Self {
            items: STM::new(IdentityMap::new(), 5, true),
        }
    }

    pub fn from_map(map: IdentityMap<LazyItem<T>>) -> Self {
        Self {
            items: STM::new(map, 1, true),
        }
    }

    /// Inserts a new item into the map
    ///
    /// Overwrites an existing item if the key already exists
    pub fn insert(&self, key: IdentityMapKey, value: LazyItem<T>) {
        let mut arc = self.items.clone();

        arc.transactional_update(|set| {
            let mut map = set.clone();
            map.insert(key.clone(), value.clone());
            map
        })
        .unwrap();
    }

    /// Inserts a new key value pair into the map if the key does not already exist
    ///
    /// Return the new value if the key did not exist, otherwise return the existing value
    ///
    /// Note: Concurrent updates should use checked_insert to avoid overwriting
    /// updates from other threads.
    pub fn checked_insert(&self, key: IdentityMapKey, default: LazyItem<T>) -> Option<LazyItem<T>> {
        let mut arc = self.items.clone();

        let (_, new_child) = arc.arcshift.rcu_project(
            |map| {
                (!map.contains(&key))
                    .then(|| {
                        let mut new_map = map.clone();
                        new_map.insert(key.clone(), default.clone());
                        Some(new_map)
                    })
                    .flatten()
            },
            |item| item.get(&key).map(|item| item.clone()),
        );

        new_child
    }

    pub fn get(&self, key: &IdentityMapKey) -> Option<LazyItem<T>> {
        let mut arc = self.items.clone();

        arc.get().get(key).cloned()
    }

    pub fn is_empty(&self) -> bool {
        let mut arc = self.items.clone();
        arc.get().is_empty()
    }

    pub fn len(&self) -> usize {
        let mut arc = self.items.clone();
        arc.get().len()
    }
}

impl<T: Clone + 'static> LazyItemVec<T> {
    pub fn new() -> Self {
        Self {
            items: STM::new(Vec::new(), 1, true),
        }
    }

    pub fn from_vec(vec: Vec<LazyItem<T>>) -> Self {
        Self {
            items: STM::new(vec, 1, true),
        }
    }

    pub fn push(&self, item: LazyItem<T>) {
        let mut items = self.items.clone();
        items
            .transactional_update(|old| {
                let mut new = old.clone();
                new.push(item.clone());
                new
            })
            .unwrap();
    }

    pub fn pop(&self) -> Option<LazyItem<T>> {
        let mut return_value = None;
        let mut items = self.items.clone();

        items
            .transactional_update(|old| {
                let mut new = old.clone();
                return_value = new.pop();
                new
            })
            .unwrap();

        return_value
    }

    pub fn get(&self, index: usize) -> Option<LazyItem<T>> {
        let mut items = self.items.clone();
        items.get().get(index).cloned()
    }

    pub fn last(&self) -> Option<LazyItem<T>> {
        let mut items = self.items.clone();
        items.get().last().cloned()
    }

    pub fn iter(&self) -> impl Iterator<Item = LazyItem<T>> {
        let mut items = self.items.clone();
        items.get().clone().into_iter()
    }

    pub fn is_empty(&self) -> bool {
        let mut items = self.items.clone();
        items.get().is_empty()
    }

    pub fn len(&self) -> usize {
        let mut items = self.items.clone();
        items.get().len()
    }

    pub fn remove(&self, index: usize) -> Option<LazyItem<T>> {
        let mut return_value = None;
        let mut items = self.items.clone();

        items
            .transactional_update(|old| {
                let mut new = old.clone();
                if index < new.len() {
                    return_value = Some(new.remove(index));
                } else {
                    return_value = None;
                }
                new
            })
            .unwrap();

        return_value
    }

    pub fn insert(&self, index: usize, item: LazyItem<T>) {
        let mut items = self.items.clone();

        items
            .transactional_update(|old| {
                let mut new = old.clone();
                new.insert(index, item.clone());
                new
            })
            .unwrap();
    }

    pub fn clear(&self) {
        let mut items = self.items.clone();
        items.transactional_update(|_| Vec::new()).unwrap();
    }
}

impl<T: Clone + 'static, const N: usize> LazyItemArray<T, N> {
    pub fn new() -> Self {
        let arr: [Option<LazyItem<T>>; N] = std::array::from_fn(|_| None);
        Self {
            items: STM::new(arr, 1, true),
        }
    }

    pub fn from_vec(vec: Vec<Option<LazyItem<T>>>) -> Self {
        let arr = LazyItemArray::new();
        let _ = vec.iter().enumerate().map(|(index, value)| {
            match value {
                Some(val) => arr.insert(index, val.clone()),
                None => {}
            };
        });
        return arr;
    }

    pub fn insert(&self, index: usize, value: LazyItem<T>) {
        let mut arc = self.items.clone();

        arc.transactional_update(|set| {
            let mut arr = set.clone();
            arr[index] = Some(value.clone());
            arr
        })
        .unwrap();
    }

    pub fn checked_insert(&self, index: usize, value: LazyItem<T>) -> Option<LazyItem<T>> {
        let mut arc = self.items.clone();

        let (_, new_child) = arc.arcshift.rcu_project(
            |arr| {
                (arr.get(index).unwrap().is_none())
                    .then(|| {
                        let mut new_arr = arr.clone();
                        new_arr[index] = Some(value.clone());
                        Some(new_arr)
                    })
                    .flatten()
            },
            |item| item.get(index).map(|item| item.clone()),
        );

        new_child.flatten()
    }

    pub fn get(&self, index: usize) -> Option<LazyItem<T>> {
        let mut arc = self.items.clone();
        arc.get().get(index).cloned().flatten()
    }

    pub fn is_empty(&mut self) -> bool {
        let mut arc = self.items.clone();
        arc.get().iter().all(Option::is_none)
    }
}
