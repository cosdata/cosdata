use super::identity_collections::{Identifiable, IdentityMap, IdentityMapKey, IdentitySet};
use super::serializer::CustomSerialize;
use super::types::{FileOffset, STM};
use super::versioning::*;
use arcshift::ArcShift;
use std::fmt;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn largest_power_of_4_below(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        let msb_position = 31 - x.leading_zeros();
        msb_position / 2
    }
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
pub enum LazyItem<T: Clone + 'static> {
    Valid {
        data: Option<ArcShift<T>>,
        file_index: ArcShift<Option<FileIndex>>,
        decay_counter: usize,
        persist_flag: Arc<AtomicBool>,
        versions: LazyItemMap<T>,
        version_id: Hash,
    },
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
                write!(f, "FileIndex(offset: {}, version: {})", offset, **version)
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
    pub items: STM<IdentityMap<LazyItem<T>>>,
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
            versions: LazyItemMap::new(),
            version_id,
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
            versions: LazyItemMap::new(),
            version_id,
        }
    }

    pub fn from_arcshift(version_id: Hash, item: ArcShift<T>) -> Self {
        Self::Valid {
            data: Some(item),
            file_index: ArcShift::new(None),
            decay_counter: 0,
            persist_flag: Arc::new(AtomicBool::new(true)),
            versions: LazyItemMap::new(),
            version_id,
        }
    }

    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid)
    }

    pub fn get_data(&self) -> Option<ArcShift<T>> {
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
        branch_id: BranchId,
        version: u32,
        lazy_item: LazyItem<T>,
    ) -> lmdb::Result<()> {
        if let Self::Valid {
            versions,
            version_id,
            ..
        } = self
        {
            let branch_last_4_bytes = (*branch_id & 0xFFFFFFFF) as u32;
            let current_version = **version_id ^ branch_last_4_bytes;
            let target_diff = version - current_version;
            let index = largest_power_of_4_below(target_diff);
            if let Some(existing_version) = versions.get(&IdentityMapKey::Int(index)) {
                return existing_version.add_version(branch_id, version, lazy_item);
            } else {
                versions.insert(IdentityMapKey::Int(index), lazy_item);
            }
        }

        Ok(())
    }

    pub fn get_versions(&self) -> Option<LazyItemMap<T>> {
        if let Self::Valid { versions, .. } = self {
            Some(versions.clone())
        } else {
            None
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
                versions: LazyItemMap::new(),
                version_id,
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
                versions: LazyItemMap::new(),
                version_id,
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
            let (file_index, decay_counter, persist_flag, version_id, versions) =
                if let LazyItem::Valid {
                    file_index,
                    decay_counter,
                    persist_flag,
                    version_id,
                    versions,
                    ..
                } = item
                {
                    (
                        file_index.clone(),
                        *decay_counter,
                        persist_flag.clone(),
                        *version_id,
                        versions.clone(),
                    )
                } else {
                    (
                        ArcShift::new(None),
                        0,
                        Arc::new(AtomicBool::new(true)),
                        0.into(),
                        LazyItemMap::new(),
                    )
                };

            LazyItem::Valid {
                data: Some(ArcShift::new(new_data)),
                file_index,
                decay_counter,
                persist_flag,
                versions,
                version_id,
            }
        });
    }

    pub fn set_file_index(&self, new_offset: Option<FileIndex>) {
        let mut arc = self.item.clone();

        arc.rcu(|item| {
            let (data, decay_counter, persist_flag, version_id, versions) =
                if let LazyItem::Valid {
                    data,
                    decay_counter,
                    persist_flag,
                    version_id,
                    versions,
                    ..
                } = item
                {
                    (
                        data.clone(),
                        *decay_counter,
                        persist_flag.clone(),
                        *version_id,
                        versions.clone(),
                    )
                } else {
                    (
                        None,
                        0,
                        Arc::new(AtomicBool::new(true)),
                        0.into(),
                        LazyItemMap::new(),
                    )
                };

            LazyItem::Valid {
                data,
                file_index: ArcShift::new(new_offset),
                decay_counter,
                persist_flag,
                versions,
                version_id,
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
            items: STM::new(IdentityMap::new(), 1, true),
        }
    }

    pub fn from_map(map: IdentityMap<LazyItem<T>>) -> Self {
        Self {
            items: STM::new(map, 1, true),
        }
    }

    pub fn insert(&self, key: IdentityMapKey, value: LazyItem<T>) {
        let mut arc = self.items.clone();

        arc.transactional_update(|set| {
            let mut map = set.clone();
            map.insert(key.clone(), value.clone());
            map
        })
        .unwrap();
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
