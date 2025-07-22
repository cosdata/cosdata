#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::{
    fmt::Debug,
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use serde::{Deserialize, Serialize};

use crate::{
    indexes::hnsw::offset_counter::IndexFileId,
    models::{
        buffered_io::BufIoError,
        cache_loader::{HNSWIndexCache, InvertedIndexCache, TFIDFIndexCache},
        inverted_index::InvertedIndexNodeData,
        prob_node::ProbNode,
        tf_idf_index::TFIDFIndexNodeData,
        types::FileOffset,
    },
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct FileIndex<FileId> {
    pub offset: FileOffset,
    pub file_id: FileId,
}

pub struct LazyItem<T, FileId> {
    data: AtomicPtr<T>,
    pub file_index: FileIndex<FileId>,
}

impl<T: PartialEq, FileId: PartialEq> PartialEq for LazyItem<T, FileId> {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            self.data.load(Ordering::Relaxed).as_ref()
                == other.data.load(Ordering::Relaxed).as_ref()
                && self.file_index == other.file_index
        }
    }
}

impl<T: Debug, FileId: Debug> Debug for LazyItem<T, FileId> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyItem")
            .field("data", unsafe {
                &self.data.load(Ordering::Relaxed).as_ref()
            })
            .field("file_index", &self.file_index)
            .finish()
    }
}

#[allow(unused)]
impl<T, FileId> LazyItem<T, FileId> {
    pub fn new(data: T, file_id: FileId, offset: FileOffset) -> *mut Self {
        let mut boxed = Box::new(Self {
            data: AtomicPtr::new(ptr::null_mut()),
            file_index: FileIndex { offset, file_id },
        });

        let data_ptr = Box::into_raw(Box::new(data));
        boxed.data.store(data_ptr, Ordering::Release);

        Box::into_raw(boxed)
    }

    pub fn new_pending(file_index: FileIndex<FileId>) -> *mut Self {
        Box::into_raw(Box::new(Self {
            data: AtomicPtr::new(ptr::null_mut()),
            file_index,
        }))
    }

    pub fn unsafe_get_data(&self) -> Option<&T> {
        // SAFETY: caller must make sure the data is not dropped by some other thread
        unsafe { self.data.load(Ordering::Acquire).as_ref() }
    }

    pub fn set_data(&self, new_data: T) {
        let old_data = self
            .data
            .swap(Box::into_raw(Box::new(new_data)), Ordering::SeqCst);
        if !old_data.is_null() {
            unsafe {
                // SAFETY: state must be a valid pointer
                drop(Box::from_raw(old_data));
            }
        }
    }

    pub fn clear_data(&self) {
        let old_data = self.data.swap(ptr::null_mut(), Ordering::SeqCst);
        if !old_data.is_null() {
            unsafe {
                drop(Box::from_raw(old_data));
            }
        }
    }

    pub fn is_ready(&self) -> bool {
        !self.data.load(Ordering::Acquire).is_null()
    }

    pub fn is_pending(&self) -> bool {
        self.data.load(Ordering::Acquire).is_null()
    }

    pub fn get_lazy_data<'a>(&self) -> Option<&'a T> {
        unsafe { self.data.load(Ordering::Acquire).as_ref() }
    }
}

impl LazyItem<ProbNode, IndexFileId> {
    pub fn try_get_data<'a>(&self, cache: &HNSWIndexCache) -> Result<&'a ProbNode, BufIoError> {
        unsafe {
            if let Some(data) = self.data.load(Ordering::Relaxed).as_ref() {
                return Ok(data);
            }
            (*(cache.get_object(self.file_index)?)).try_get_data(cache)
        }
    }
}

impl LazyItem<InvertedIndexNodeData, ()> {
    pub fn try_get_data<'a>(
        &self,
        cache: &InvertedIndexCache,
    ) -> Result<&'a InvertedIndexNodeData, BufIoError> {
        unsafe {
            if let Some(data) = self.data.load(Ordering::Relaxed).as_ref() {
                return Ok(data);
            }

            let offset = self.file_index.offset;
            (*(cache.get_data(offset)?)).try_get_data(cache)
        }
    }
}

impl LazyItem<TFIDFIndexNodeData, ()> {
    pub fn try_get_data<'a>(
        &self,
        cache: &TFIDFIndexCache,
    ) -> Result<&'a TFIDFIndexNodeData, BufIoError> {
        unsafe {
            if let Some(data) = self.data.load(Ordering::Relaxed).as_ref() {
                return Ok(data);
            }

            let offset = self.file_index.offset;
            (*(cache.get_data(offset)?)).try_get_data(cache)
        }
    }
}

impl<T, FileId> Drop for LazyItem<T, FileId> {
    fn drop(&mut self) {
        let data_ptr = self.data.load(Ordering::SeqCst);
        unsafe {
            if !data_ptr.is_null() {
                drop(Box::from_raw(data_ptr));
            }
        }
    }
}
