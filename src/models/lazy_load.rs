use super::cache_loader::{Cacheable, NodeRegistry};
use super::common::WaCustomError;
use super::identity_collections::{Identifiable, IdentityMap, IdentityMapKey, IdentitySet};
use super::serializer::CustomSerialize;
use super::types::{FileOffset, STM};
use super::versioning::*;
use arcshift::ArcShift;
use core::panic;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub fn largest_power_of_4_below(x: u16) -> u8 {
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

    // must be small enough to fit inside u8
    let msb_position = (15 - x.leading_zeros()) as u8; // Find the most significant bit's position
    msb_position / 2 // Return the power index of the largest 4^n ≤ x
}

pub trait SyncPersist {
    fn set_persistence(&self, flag: bool);
    fn needs_persistence(&self) -> bool;
    fn get_current_version(&self) -> Hash;
    fn get_current_version_number(&self) -> u16;
}

pub const CHUNK_SIZE: usize = 5;

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone, Serialize, Deserialize)]
pub enum FileIndex {
    Valid {
        offset: FileOffset,
        version_number: u16,
        version_id: Hash,
    },
    Invalid,
}

impl FileIndex {
    pub fn get_offset(&self) -> Option<FileOffset> {
        match self {
            Self::Invalid => None,
            Self::Valid { offset, .. } => Some(*offset),
        }
    }

    pub fn get_version_number(&self) -> Option<u16> {
        match self {
            Self::Invalid => None,
            Self::Valid { version_number, .. } => Some(*version_number),
        }
    }

    pub fn get_version_id(&self) -> Option<Hash> {
        match self {
            Self::Invalid => None,
            Self::Valid { version_id, .. } => Some(*version_id),
        }
    }
}

#[derive(Clone)]
// As the name suggests, this is a wrapper for lazy-loading the inner data. Its
// serialization/deserialization mechanisms are designed to handle cyclic data while minimizing
// redundant re-serializations, ensuring optimal performance.
pub enum LazyItem<T: Clone + 'static> {
    Valid {
        // Holds the actual data. Wrapped in an `Option` to indicate whether the data is loaded.
        data: ArcShift<Option<Arc<T>>>,
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
        // Current version
        version_number: u16,
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
            FileIndex::Valid {
                offset,
                version_number,
                version_id,
            } => {
                write!(
                    f,
                    "FileIndex(offset: {}, version_number: {}, version_id: {})",
                    offset.0, version_number, **version_id
                )
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

            let mut data_arc = data.clone();

            if let Some(data) = data_arc.get() {
                return LazyItemId::Memory(data.get_id());
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
    pub serialized_offset: ArcShift<Option<u32>>,
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

#[derive(Clone)]
pub struct VectorData {
    pub data: Box<[u32; 64]>,
    pub is_serialized: bool,
}

#[derive(Clone)]
pub struct IncrementalSerializableGrowableData {
    pub items: LazyItemVec<STM<VectorData>>,
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

    fn get_current_version_number(&self) -> u16 {
        if let Self::Valid { version_number, .. } = self {
            *version_number
        } else {
            0
        }
    }
}

impl<T: Clone + 'static> LazyItem<T> {
    pub fn new(version_id: Hash, version_number: u16, item: T) -> Self {
        Self::Valid {
            data: ArcShift::new(Some(Arc::new(item))),
            file_index: ArcShift::new(None),
            decay_counter: 0,
            persist_flag: Arc::new(AtomicBool::new(true)),
            versions: LazyItemVec::new(),
            version_id,
            version_number,
            serialized_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn new_invalid() -> Self {
        Self::Invalid
    }

    pub fn from_data(version_id: Hash, version_number: u16, data: T) -> Self {
        LazyItem::Valid {
            data: ArcShift::new(Some(Arc::new(data))),
            file_index: ArcShift::new(None),
            decay_counter: 0,
            persist_flag: Arc::new(AtomicBool::new(true)),
            versions: LazyItemVec::new(),
            version_id,
            version_number,
            serialized_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn from_arc(version_id: Hash, version_number: u16, item: Arc<T>) -> Self {
        Self::Valid {
            data: ArcShift::new(Some(item)),
            file_index: ArcShift::new(None),
            decay_counter: 0,
            persist_flag: Arc::new(AtomicBool::new(true)),
            versions: LazyItemVec::new(),
            version_id,
            version_number,
            serialized_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid)
    }

    pub fn get_lazy_data(&self) -> Option<ArcShift<Option<Arc<T>>>> {
        if let Self::Valid { data, .. } = self {
            return Some(data.clone());
        }
        None
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

    pub fn get_versions(&self) -> Option<LazyItemVec<T>> {
        if let Self::Valid { versions, .. } = self {
            Some(versions.clone())
        } else {
            None
        }
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

impl<T: Clone + CustomSerialize + Cacheable + 'static> LazyItem<T> {
    pub fn add_version(&self, cache: Arc<NodeRegistry>, lazy_item: LazyItem<T>) {
        let latest_local_version_number = self.get_latest_version(cache.clone()).1;
        self.add_version_inner(cache, lazy_item, 0, latest_local_version_number + 1)
    }

    fn add_version_inner(
        &self,
        cache: Arc<NodeRegistry>,
        lazy_item: LazyItem<T>,
        self_relative_version_number: u16,
        target_relative_version_number: u16,
    ) {
        if let Self::Valid {
            data,
            file_index,
            versions,
            ..
        } = self
        {
            let mut data = data.clone();
            let mut versions = versions.clone();
            let target_diff = target_relative_version_number - self_relative_version_number;
            let index = largest_power_of_4_below(target_diff);

            if data.get().is_none() {
                let Some(file_index) = file_index.clone().get().clone() else {
                    unreachable!("data and file_index both cannot be None at the time!");
                };
                let item: LazyItem<T> = cache
                    .clone()
                    .load_item(file_index)
                    .expect("Deserialization failed");
                let (deserialized_data, deserialized_versions) = match item {
                    LazyItem::Valid { data, versions, .. } => (
                        data.clone().get().clone().unwrap(),
                        versions.items.clone().get().clone(),
                    ),
                    LazyItem::Invalid => {
                        unreachable!("Deserialized LazyItem should not be Invalid")
                    }
                };
                versions.items.update(deserialized_versions);
                data.update(Some(deserialized_data));
            };

            if let Some(existing_version) = versions.get(index as usize) {
                return existing_version.add_version_inner(
                    cache,
                    lazy_item,
                    self_relative_version_number + (1 << (2 * index)),
                    target_relative_version_number,
                );
            }

            versions.insert(index as usize, lazy_item);
        }
    }

    pub fn get_version(&self, cache: Arc<NodeRegistry>, version: u16) -> Option<LazyItem<T>> {
        match self {
            Self::Valid {
                data,
                file_index,
                versions,
                version_number,
                ..
            } => {
                if &version < version_number {
                    return None;
                }
                if &version == version_number {
                    return Some(self.clone());
                }
                let mut data = data.clone();
                let mut versions = versions.clone();
                if data.is_none() {
                    let Some(file_index) = file_index.clone().get().clone() else {
                        unreachable!("data and file_index both cannot be None at the time!");
                    };
                    let item: LazyItem<T> = cache
                        .clone()
                        .load_item(file_index)
                        .expect("Deserialization failed");
                    let (deserialized_data, deserialized_versions) = match item {
                        LazyItem::Valid { data, versions, .. } => (
                            data.clone().get().clone().unwrap(),
                            versions.items.clone().get().clone(),
                        ),
                        LazyItem::Invalid => {
                            unreachable!("Deserialized LazyItem should not be Invalid")
                        }
                    };
                    versions.items.update(deserialized_versions);
                    data.update(Some(deserialized_data));
                };
                let Some(mut prev) = versions.get(0) else {
                    return None;
                };
                let mut i = 1;
                while let Some(next) = versions.get(i) {
                    if version < next.get_current_version_number() {
                        return prev.get_version(cache, version);
                    }
                    prev = next;
                    i += 1;
                }

                prev.get_version(cache, version)
            }
            Self::Invalid => None,
        }
    }

    // returns (latest version of current node, relative version number)
    pub fn get_latest_version(&self, cache: Arc<NodeRegistry>) -> (LazyItem<T>, u16) {
        if let Self::Valid {
            versions,
            data,
            file_index,
            ..
        } = self
        {
            let mut data = data.clone();
            let mut versions = versions.clone();
            if data.is_none() {
                let Some(file_index) = file_index.clone().get().clone() else {
                    unreachable!("data and file_index both cannot be None at the time!");
                };
                let item: LazyItem<T> = cache
                    .clone()
                    .load_item(file_index)
                    .expect("Deserialization failed");
                let (deserialized_data, deserialized_versions) = match item {
                    LazyItem::Valid { data, versions, .. } => (
                        data.clone().get().clone().unwrap(),
                        versions.items.clone().get().clone(),
                    ),
                    LazyItem::Invalid => {
                        unreachable!("Deserialized LazyItem should not be Invalid")
                    }
                };
                versions.items.update(deserialized_versions);
                data.update(Some(deserialized_data));
            };
            if let Some(last) = versions.items.get().last() {
                let (latest_version, relative_local_version_number) =
                    last.get_latest_version(cache);
                return (
                    latest_version,
                    (1u16 << ((versions.len() as u8 - 1) * 2)) + relative_local_version_number,
                );
            }
        };
        (self.clone(), 0)
    }

    pub fn try_get_data(&self, cache: Arc<NodeRegistry>) -> Result<Arc<T>, WaCustomError> {
        if let Self::Valid {
            data,
            file_index,
            versions,
            ..
        } = self
        {
            let mut data_arc = data.clone();
            if let Some(data) = data_arc.get() {
                return Ok(data.clone());
            }

            let mut file_index_arc = file_index.clone();
            let offset = file_index_arc
                .get()
                .as_ref()
                .ok_or(WaCustomError::LazyLoadingError(
                    "LazyItem offset is None".to_string(),
                ))?;

            let item: LazyItem<T> = LazyItem::deserialize(
                cache.get_bufmans(),
                offset.clone(),
                cache,
                1000,
                &mut HashSet::new(),
            )
            .map_err(|e| WaCustomError::BufIo(Arc::new(e)))?;

            let (data, deserialized_versions) = match item {
                Self::Valid {
                    mut data,
                    mut versions,
                    ..
                } => (
                    data.get().clone().ok_or(WaCustomError::LazyLoadingError(
                        "Deserialized LazyItem is None".to_string(),
                    ))?,
                    versions.items.get().clone(),
                ),
                Self::Invalid => {
                    return Err(WaCustomError::LazyLoadingError(
                        "Deserialized LazyItem is invalid".to_string(),
                    ));
                }
            };

            data_arc.update(Some(data.clone()));

            let mut version_arc = versions.clone();
            version_arc.items.update(deserialized_versions);

            Ok(data)
        } else {
            return Err(WaCustomError::LazyLoadingError(
                "LazyItem is invalid".to_string(),
            ));
        }
    }

    pub fn get_data(&self, cache: Arc<NodeRegistry>) -> Arc<T> {
        if let Self::Valid {
            data,
            file_index,
            versions,
            ..
        } = self
        {
            let mut data_arc = data.clone();
            if let Some(data) = data_arc.get() {
                return data.clone();
            }

            let file_index = file_index
                .clone()
                .get()
                .clone()
                .expect("LazyItem offset is None");

            // let deserialized = cache.load_item(file_index).expect("Deserialization error");
            let item: LazyItem<T> = LazyItem::deserialize(
                cache.get_bufmans(),
                file_index,
                cache,
                1000,
                &mut HashSet::new(),
            )
            .expect("Deserialization failed");

            let (data, deserialized_versions) = match item {
                Self::Valid {
                    mut data,
                    mut versions,
                    ..
                } => (
                    data.get().clone().expect("Deserialized LazyItem is None"),
                    versions.items.get().clone(),
                ),
                Self::Invalid => panic!("Deserialized LazyItem is Invalid"),
            };

            data_arc.update(Some(data.clone()));

            let mut version_arc = versions.clone();
            version_arc.items.update(deserialized_versions);

            data
        } else {
            panic!("LazyItem is invalid");
        }
    }
}

impl<T: Clone + 'static> LazyItemRef<T> {
    pub fn new(version_id: Hash, version_number: u16, item: T) -> Self {
        Self {
            item: ArcShift::new(LazyItem::Valid {
                data: ArcShift::new(Some(Arc::new(item))),
                file_index: ArcShift::new(None),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions: LazyItemVec::new(),
                version_id,
                version_number,
                serialized_flag: Arc::new(AtomicBool::new(false)),
            }),
        }
    }

    pub fn new_invalid() -> Self {
        Self {
            item: ArcShift::new(LazyItem::Invalid),
        }
    }

    pub fn from_arc(version_id: Hash, version_number: u16, item: Arc<T>) -> Self {
        Self {
            item: ArcShift::new(LazyItem::Valid {
                data: ArcShift::new(Some(item)),
                file_index: ArcShift::new(None),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions: LazyItemVec::new(),
                version_id,
                version_number,
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

    pub fn get_lazy_data(&self) -> Option<ArcShift<Option<Arc<T>>>> {
        let mut arc = self.item.clone();
        if let LazyItem::Valid { data, .. } = arc.get() {
            return Some(data.clone());
        }
        None
    }

    pub fn set_data(&self, new_data: T) {
        let mut arc = self.item.clone();

        arc.rcu(|item| {
            let (
                file_index,
                decay_counter,
                persist_flag,
                version_id,
                version_number,
                versions,
                serialized_flag,
            ) = if let LazyItem::Valid {
                file_index,
                decay_counter,
                persist_flag,
                version_id,
                version_number,
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
                    *version_number,
                    versions.clone(),
                    serialized_flag.clone(),
                )
            } else {
                (
                    ArcShift::new(None),
                    0,
                    Arc::new(AtomicBool::new(true)),
                    0.into(),
                    0,
                    LazyItemVec::new(),
                    Arc::new(AtomicBool::new(false)),
                )
            };

            LazyItem::Valid {
                data: ArcShift::new(Some(Arc::new(new_data))),
                file_index,
                decay_counter,
                persist_flag,
                versions,
                version_id,
                version_number,
                serialized_flag,
            }
        });
    }

    pub fn set_file_index(&self, new_offset: Option<FileIndex>) {
        let mut arc = self.item.clone();

        arc.rcu(|item| {
            let (
                data,
                decay_counter,
                persist_flag,
                version_id,
                version_number,
                versions,
                serialized_flag,
            ) = if let LazyItem::Valid {
                data,
                decay_counter,
                persist_flag,
                version_id,
                version_number,
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
                    *version_number,
                    versions.clone(),
                    serialized_flag.clone(),
                )
            } else {
                (
                    ArcShift::new(None),
                    0,
                    Arc::new(AtomicBool::new(true)),
                    0.into(),
                    0,
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
                version_number,
                serialized_flag,
            }
        });
    }

    pub fn get_current_version(&self) -> Hash {
        let mut arc = self.item.clone();
        arc.get().get_current_version()
    }

    pub fn get_current_version_number(&self) -> u16 {
        let mut arc = self.item.clone();
        arc.get().get_current_version_number()
    }
}

impl<T, E> EagerLazyItemSet<T, E>
where
    T: Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    pub fn new() -> Self {
        Self {
            serialized_offset: ArcShift::new(None),
            items: STM::new(IdentitySet::new(), 5, false),
        }
    }

    pub fn from_set(set: IdentitySet<EagerLazyItem<T, E>>) -> Self {
        Self {
            serialized_offset: ArcShift::new(None),
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
            items: STM::new(Vec::new(), 4, true),
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

    pub fn resize(&self, new_len: usize, value: LazyItem<T>) {
        let mut items = self.items.clone();
        items
            .transactional_update(|old| {
                let mut new = old.clone();
                new.resize(new_len, value.clone());
                new
            })
            .unwrap();
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

impl VectorData {
    pub fn new() -> Self {
        Self {
            data: Box::new([u32::MAX; 64]),
            is_serialized: false,
        }
    }

    pub fn from_array(data: [u32; 64], is_serialized: bool) -> Self {
        Self {
            data: Box::new(data),
            is_serialized,
        }
    }

    pub fn is_serialized(&self) -> bool {
        self.is_serialized
    }

    pub fn get(&self, index: usize) -> Option<u32> {
        if index < 64 {
            match self.data[index] {
                u32::MAX => None,
                val => Some(val),
            }
        } else {
            None
        }
    }

    pub fn set(&mut self, index: usize, vec_id: u32) {
        if index < 64 {
            self.data[index] = vec_id;
            self.is_serialized = false;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&x| x == u32::MAX)
    }
}

impl IncrementalSerializableGrowableData {
    pub fn new() -> Self {
        Self {
            items: LazyItemVec::new(),
        }
    }

    pub fn from_vec(vec: Vec<VectorData>) -> Self {
        let items = vec
            .iter()
            .map(|data| {
                LazyItem::new(
                    Hash::from(u32::MAX),
                    u16::MAX,
                    STM::new(
                        VectorData::from_array(*data.data, data.is_serialized),
                        1,
                        true,
                    ),
                )
            })
            .collect();
        Self {
            items: LazyItemVec::from_vec(items),
        }
    }

    pub fn insert(&mut self, vec_id: u32) {
        let insert_dimension = (vec_id % 64) as usize;
        let insert_index = (vec_id / 64) as usize;
        let items_len = self.items.len();

        if items_len <= insert_index {
            self.items.resize(
                insert_index + 1,
                LazyItem::new(
                    Hash::from(u32::MAX),
                    u16::MAX,
                    STM::new(VectorData::new(), 1, true),
                ),
            );
        }

        let mut vector_data_arcshift = self
            .items
            .get(insert_index)
            .unwrap()
            .get_lazy_data()
            .unwrap();
        let mut vector_data_stm = (*vector_data_arcshift.get().clone().unwrap()).clone();
        vector_data_stm
            .transactional_update(|old| {
                let mut new = old.clone();
                new.set(insert_dimension, vec_id);
                new
            })
            .unwrap();
    }

    pub fn get(&self, vec_id: u32) -> Option<u32> {
        let insert_dimension = (vec_id % 64) as usize;
        let insert_index = (vec_id / 64) as usize;

        if self.items.len() <= insert_index {
            return None;
        }

        let vector_data_lazy_item = self.items.get(insert_index).unwrap();
        let mut vector_data_stm = (*vector_data_lazy_item
            .get_lazy_data()
            .unwrap()
            .get()
            .clone()
            .unwrap())
        .clone();
        vector_data_stm.get().get(insert_dimension)
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use tempfile::tempdir;

    use crate::models::buffered_io::BufferManagerFactory;

    use super::*;

    #[test]
    fn test_lazy_item_versions_add_and_get() {
        let temp_dir = tempdir().unwrap();
        let bufmans = Arc::new(BufferManagerFactory::new(
            temp_dir.as_ref().into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            1.0,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        let root = LazyItem::new(Hash::from(0), 0, 0.0);

        for i in 1..=u16::MAX {
            let version = LazyItem::new(Hash::from(0), i, 0.0);
            root.add_version(cache.clone(), version);
        }

        for i in 0..=u16::MAX {
            assert_eq!(
                root.get_version(cache.clone(), i)
                    .unwrap()
                    .get_current_version_number(),
                i
            );
        }
    }

    #[test]
    fn test_lazy_item_versions_add_and_get_with_skipped_items() {
        let temp_dir = tempdir().unwrap();
        let bufmans = Arc::new(BufferManagerFactory::new(
            temp_dir.as_ref().into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            1.0,
        ));
        let cache = Arc::new(NodeRegistry::new(1000, bufmans));
        let root = LazyItem::new(Hash::from(0), 0, 0.0);
        let mut i = 0;
        let mut rng = thread_rng();
        let mut versions = vec![0];

        while i < u16::MAX - 100 {
            i += rng.gen_range(3..100);

            let version = LazyItem::new(Hash::from(0), i, 0.0);
            root.add_version(cache.clone(), version);
            versions.push(i);
        }

        for i in versions {
            assert_eq!(
                root.get_version(cache.clone(), i)
                    .unwrap()
                    .get_current_version_number(),
                i
            );
        }
    }
}
