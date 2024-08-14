use super::identity_collections::{Identifiable, IdentityMap, IdentityMapKey, IdentitySet};
use super::serializer::CustomSerialize;
use super::types::{FileOffset, STM};
use arcshift::ArcShift;
use std::hash::Hash;

pub trait SyncPersist {
    fn set_persistence(&self, flag: bool);
    fn needs_persistence(&self) -> bool;
}

pub const CHUNK_SIZE: usize = 5;

#[derive(Clone)]
pub enum LazyItem<T: Clone + 'static> {
    Valid {
        data: Option<ArcShift<T>>,
        offset: ArcShift<Option<FileOffset>>,
        decay_counter: usize,
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
    Persist(u32),
}

impl<T> Identifiable for LazyItem<T>
where
    T: Clone + Identifiable<Id = u64> + 'static,
{
    type Id = LazyItemId;

    fn get_id(&self) -> Self::Id {
        if let LazyItem::Valid { data, offset, .. } = self {
            if let Some(offset) = offset.clone().get().clone() {
                return LazyItemId::Persist(offset);
            }

            if let Some(data) = data {
                let mut arc = data.clone();
                return LazyItemId::Memory(arc.get().get_id());
            }
        }

        LazyItemId::Persist(u32::MAX)
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

impl<T: Clone + 'static> LazyItem<T> {
    pub fn new(item: T) -> Self {
        Self::Valid {
            data: Some(ArcShift::new(item)),
            offset: ArcShift::new(None),
            decay_counter: 0,
        }
    }

    pub fn new_invalid() -> Self {
        Self::Invalid
    }

    pub fn from_data(data: T) -> Self {
        LazyItem::Valid {
            data: Some(ArcShift::new(data)),
            offset: ArcShift::new(None),
            decay_counter: 0,
        }
    }

    pub fn from_arcshift(item: ArcShift<T>) -> Self {
        Self::Valid {
            data: Some(item),
            offset: ArcShift::new(None),
            decay_counter: 0,
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

    pub fn get_offset(&self) -> Option<FileOffset> {
        if let Self::Valid { offset, .. } = self {
            return offset.clone().get().clone();
        }
        None
    }

    pub fn set_offset(&self, new_offset: Option<FileOffset>) {
        if let Self::Valid { offset, .. } = self {
            offset.clone().update(new_offset);
        }
    }
}

impl<T: Clone + 'static> LazyItemRef<T> {
    pub fn new(item: T) -> Self {
        Self {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(ArcShift::new(item)),
                offset: ArcShift::new(None),
                decay_counter: 0,
            }),
        }
    }

    pub fn new_invalid() -> Self {
        Self {
            item: ArcShift::new(LazyItem::Invalid),
        }
    }

    pub fn from_arcshift(item: ArcShift<T>) -> Self {
        Self {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(item),
                offset: ArcShift::new(None),
                decay_counter: 0,
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
            let (offset, decay_counter) = if let LazyItem::Valid {
                offset,
                decay_counter,
                ..
            } = item
            {
                (offset.clone(), *decay_counter)
            } else {
                (ArcShift::new(None), 0)
            };
            LazyItem::Valid {
                data: Some(ArcShift::new(new_data)),
                offset,
                decay_counter,
            }
        });
    }

    pub fn set_offset(&self, new_offset: Option<FileOffset>) {
        let mut arc = self.item.clone();

        arc.rcu(|item| {
            let (data, decay_counter) = if let LazyItem::Valid {
                data,
                decay_counter,
                ..
            } = item
            {
                (data.clone(), *decay_counter)
            } else {
                (None, 0)
            };
            LazyItem::Valid {
                data,
                offset: ArcShift::new(new_offset),
                decay_counter,
            }
        });
    }
}

impl<T, E> EagerLazyItemSet<T, E>
where
    T: Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    pub fn new() -> Self {
        Self {
            items: STM::new(IdentitySet::new(), 8, true),
        }
    }

    pub fn from_set(set: IdentitySet<EagerLazyItem<T, E>>) -> Self {
        Self {
            items: STM::new(set, 1, true),
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
        });
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
        });
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
