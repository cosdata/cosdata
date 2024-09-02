use std::collections::HashMap;
use std::hash::Hash;

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

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.map.into_iter().map(|(_, value)| value)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn remove(&mut self, value: &T) -> Option<T> {
        self.map.remove(&value.get_id())
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

    pub fn from_iter(iter: impl Iterator<Item = (IdentityMapKey, T)>) -> Self {
        Self {
            map: iter.collect(),
        }
    }

    pub fn insert(&mut self, key: IdentityMapKey, value: T) -> Option<T> {
        self.map.insert(key, value)
    }

    pub fn get(&self, key: &IdentityMapKey) -> Option<&T> {
        self.map.get(key)
    }

    pub fn contains(&self, key: &IdentityMapKey) -> bool {
        self.map.contains_key(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IdentityMapKey, &T)> {
        self.map.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = (IdentityMapKey, T)> {
        self.map.into_iter()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn remove(&mut self, key: &IdentityMapKey) -> Option<T> {
        self.map.remove(key)
    }
}
