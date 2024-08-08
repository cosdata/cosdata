use arcshift::ArcShift;

use super::serializer::CustomSerialize;
use super::types::Item;
use std::collections::HashMap;
use std::hash::Hash;

pub trait SyncPersist {
    fn set_persistence(&self, flag: bool);
    fn needs_persistence(&self) -> bool;
}

type FileOffset = u32;
type VersionId = u16;
type HNSWLevel = u8;

pub const CHUNK_SIZE: usize = 5;

pub trait Identifiable {
    type Id: Eq + Hash;
    fn get_id(&self) -> Self::Id;
}

#[derive(Debug, Clone)]
pub struct IdentitySet<T: Identifiable> {
    map: HashMap<T::Id, T>,
}

impl<T: Identifiable> IdentitySet<T> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn from_iter(iter: impl Iterator<Item = T>) -> Self {
        Self {
            map: iter.map(|item| (item.get_id(), item)).collect(),
        }
    }

    pub fn insert(&mut self, value: T) -> Option<T> {
        self.map.insert(value.get_id(), value)
    }

    pub fn contains(&self, value: &T) -> bool {
        self.map.contains_key(&value.get_id())
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.iter().map(|(_, value)| value)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum IdentityMapKey {
    String(String),
    Int(u32),
}

#[derive(Debug, Clone)]
pub struct IdentityMap<T> {
    map: HashMap<IdentityMapKey, T>,
}

impl<T> IdentityMap<T> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: IdentityMapKey, value: T) -> Option<T> {
        self.map.insert(key, value)
    }

    pub fn contains(&self, key: &IdentityMapKey) -> bool {
        self.map.contains_key(key)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(Clone)]
pub enum LazyItem<T: Clone + 'static> {
    Valid {
        data: Option<Item<T>>,
        offset: Option<FileOffset>,
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
            if let Some(offset) = offset {
                return LazyItemId::Persist(*offset);
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
    pub item: Item<LazyItem<T>>,
}

#[derive(Clone)]
pub struct EagerLazyItemSet<T, E>
where
    T: Clone + Identifiable<Id = u64> + 'static,
    E: Clone + CustomSerialize + 'static,
{
    pub items: Item<IdentitySet<EagerLazyItem<T, E>>>,
}

#[derive(Clone)]
pub struct LazyItemSet<T>
where
    T: Clone + Identifiable<Id = u64> + 'static,
{
    pub items: Item<IdentitySet<LazyItem<T>>>,
}

#[derive(Clone)]
pub struct LazyItemMap<T: Clone + 'static> {
    pub items: Item<IdentityMap<LazyItem<T>>>,
}

impl<T: Clone> LazyItem<T> {
    pub fn from_data(data: T) -> Self {
        LazyItem::Valid {
            data: Some(Item::new(data)),
            offset: None,
            decay_counter: 0,
        }
    }
}

impl<T: Clone + 'static> LazyItem<T> {
    pub fn new(item: T) -> Self {
        Self::Valid {
            data: Some(Item::new(item)),
            offset: None,
            decay_counter: 0,
        }
    }

    pub fn new_invalid() -> Self {
        Self::Invalid
    }

    pub fn from_item(item: Item<T>) -> Self {
        Self::Valid {
            data: Some(item),
            offset: None,
            decay_counter: 0,
        }
    }

    pub fn get_data(&self) -> Option<Item<T>> {
        if let Self::Valid { data, .. } = self {
            return data.clone();
        }
        None
    }

    pub fn set_data(&mut self, new_data: T) {
        if let Self::Valid { data, .. } = self {
            *data = Some(Item::new(new_data))
        }
    }

    pub fn set_offset(&mut self, new_offset: Option<FileOffset>) {
        if let Self::Valid { offset, .. } = self {
            *offset = new_offset;
        }
    }
}

impl<T: Clone + 'static> LazyItemRef<T> {
    pub fn new(item: T) -> Self {
        LazyItemRef {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(Item::new(item)),
                offset: None,
                decay_counter: 0,
            }),
        }
    }

    pub fn new_invalid() -> Self {
        LazyItemRef {
            item: Item::new(LazyItem::Invalid),
        }
    }

    pub fn from_item(item: Item<T>) -> Self {
        LazyItemRef {
            item: ArcShift::new(LazyItem::Valid {
                data: Some(item),
                offset: None,
                decay_counter: 0,
            }),
        }
    }

    pub fn get_data(&self) -> Option<Item<T>> {
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
                (None, 0)
            };
            LazyItem::Valid {
                data: Some(Item::new(new_data)),
                offset,
                decay_counter,
            }
        });
    }

    pub fn set_offset(&mut self, new_offset: Option<FileOffset>) {
        self.item.rcu(|item| {
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
                offset: new_offset,
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
            items: Item::new(IdentitySet::new()),
        }
    }

    pub fn insert(&self, item: LazyItem<T>, eager_data: E) {
        let mut arc = self.items.clone();

        arc.rcu(|set| {
            let mut set = set.clone();
            set.insert(EagerLazyItem(eager_data, item));
            set
        })
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
            items: Item::new(IdentitySet::new()),
        }
    }

    pub fn insert(&self, item: LazyItem<T>) {
        let mut arc = self.items.clone();

        arc.rcu(|set| {
            let mut set = set.clone();
            set.insert(item);
            set
        })
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

impl<T: Clone + Identifiable<Id = u64> + 'static> LazyItemMap<T> {
    pub fn new() -> Self {
        Self {
            items: Item::new(IdentityMap::new()),
        }
    }

    pub fn insert(&self, key: IdentityMapKey, value: LazyItem<T>) {
        let mut arc = self.items.clone();

        arc.rcu(|set| {
            let mut map = set.clone();
            map.insert(key, value);
            map
        })
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
