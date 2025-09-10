use std::mem;

use super::{lazy_item::FileIndex, types::FileOffset, versioning::VersionNumber};

#[derive(Debug, PartialEq, Eq)]
pub struct OmVersionedVec<T> {
    pub serialized_at: Option<FileOffset>,
    pub version: VersionNumber,
    pub list: Vec<T>,
    pub prev: Option<LazyOmVersionedVec<T>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LazyOmVersionedVec<T> {
    Vec(Box<OmVersionedVec<T>>),
    FileIndex(FileIndex<VersionNumber>),
}

impl<T> OmVersionedVec<T> {
    pub fn new(version: VersionNumber) -> Self {
        Self {
            serialized_at: None,
            version,
            list: Vec::new(),
            prev: None,
        }
    }

    pub fn push(&mut self, version: VersionNumber, value: T) {
        if self.version != version {
            let new = Self::new(version);
            let prev = mem::replace(self, new);
            self.prev = Some(LazyOmVersionedVec::Vec(Box::new(prev)));
        }

        self.list.push(value);
    }

    pub fn push_sorted(&mut self, version: VersionNumber, value: T)
    where
        T: Ord,
    {
        if self.version != version {
            let new = Self::new(version);
            let prev = mem::replace(self, new);
            self.prev = Some(LazyOmVersionedVec::Vec(Box::new(prev)));
        }

        let mut i = self.list.len();
        while i > 0 && self.list[i - 1] > value {
            i -= 1;
        }
        self.list.insert(i, value);
    }
}

impl<T> LazyOmVersionedVec<T> {
    pub fn push(&mut self, version: VersionNumber, value: T) {
        match self {
            Self::Vec(vec) => vec.push(version, value),
            Self::FileIndex(index) => {
                let mut new = OmVersionedVec::new(version);
                new.push(version, value);
                new.prev = Some(Self::FileIndex(*index));
                *self = Self::Vec(Box::new(new));
            }
        }
    }

    pub fn push_sorted(&mut self, version: VersionNumber, value: T)
    where
        T: Ord,
    {
        match self {
            Self::Vec(vec) => vec.push_sorted(version, value),
            Self::FileIndex(index) => {
                let mut new = OmVersionedVec::new(version);
                new.push_sorted(version, value);
                new.prev = Some(Self::FileIndex(*index));
                *self = Self::Vec(Box::new(new));
            }
        }
    }

    pub fn version(&self) -> VersionNumber {
        match self {
            Self::Vec(vec) => vec.version,
            Self::FileIndex(index) => index.file_id,
        }
    }
}
