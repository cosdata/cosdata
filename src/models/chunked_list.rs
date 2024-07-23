use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

pub trait SyncPersist {
    fn set_persistence(&mut self, flag: bool);
    fn needs_persistence(&self) -> bool;
}

type FileOffset = u32;
type VersionId = u16;
type HNSWLevel = u8;

pub const CHUNK_SIZE: usize = 5;

#[derive(Debug, Clone)]
pub enum LazyItem<T: Clone> {
    Ready(Arc<T>, Option<FileOffset>),
    LazyLoad(FileOffset),
    Null,
}

#[derive(Debug, Clone)]
pub struct LazyItems<T: Clone> {
    pub items: Vec<LazyItem<T>>,
}

impl<T: Clone> LazyItem<T> {
    pub fn get_offset(&self) -> Option<FileOffset> {
        match self {
            LazyItem::Ready(_, offset) => *offset,
            LazyItem::LazyLoad(offset) => Some(*offset),
            LazyItem::Null => None,
        }
    }

    pub fn new_lazy(offset: FileOffset) -> Self {
        if offset == u32::MAX {
            LazyItem::Null
        } else {
            LazyItem::LazyLoad(offset)
        }
    }

    pub fn set_offset(&mut self, offset: Option<FileOffset>) {
        match self {
            LazyItem::Ready(_, stored_offset) => *stored_offset = offset,
            LazyItem::LazyLoad(stored_offset) => *stored_offset = offset.unwrap_or(0),
            LazyItem::Null => {}
        }
    }
}

impl<T: Clone> LazyItems<T> {
    pub fn new() -> Self {
        LazyItems { items: Vec::new() }
    }

    pub fn push(&mut self, item: LazyItem<T>) {
        self.items.push(item);
    }

    pub fn get(&self, index: usize) -> Option<&LazyItem<T>> {
        self.items.get(index)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &LazyItem<T>> {
        self.items.iter()
    }
}
