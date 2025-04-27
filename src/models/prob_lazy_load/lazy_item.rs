#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::{
    fmt::Debug,
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use serde::{Deserialize, Serialize};

use crate::models::{
    buffered_io::BufIoError,
    cache_loader::{HNSWIndexCache, InvertedIndexCache, TFIDFIndexCache},
    inverted_index::InvertedIndexNodeData,
    prob_node::ProbNode,
    tf_idf_index::TFIDFIndexNodeData,
    types::FileOffset,
    versioning::Hash,
};

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone, Serialize, Deserialize)]
pub struct FileIndex {
    pub offset: FileOffset,
    pub version_number: u16,
    pub version_id: Hash,
}

pub struct ProbLazyItem<T> {
    data: AtomicPtr<T>,
    pub file_index: FileIndex,
    pub is_level_0: bool,
}

impl<T: PartialEq> PartialEq for ProbLazyItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.is_level_0 == other.is_level_0
            && unsafe { *self.data.load(Ordering::Relaxed) == *other.data.load(Ordering::Relaxed) }
    }
}

impl<T: Debug> Debug for ProbLazyItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProbLazyItem")
            .field("data", unsafe { &*self.data.load(Ordering::Relaxed) })
            .field("file_index", &self.file_index)
            .field("is_level_0", &self.is_level_0)
            .finish()
    }
}

#[allow(unused)]
impl<T> ProbLazyItem<T> {
    pub fn new(
        data: T,
        version_id: Hash,
        version_number: u16,
        is_level_0: bool,
        offset: FileOffset,
    ) -> *mut Self {
        Box::into_raw(Box::new(Self {
            data: AtomicPtr::new(Box::into_raw(Box::new(data))),
            file_index: FileIndex {
                offset,
                version_number,
                version_id,
            },
            is_level_0,
        }))
    }

    pub fn new_pending(file_index: FileIndex, is_level_0: bool) -> *mut Self {
        Box::into_raw(Box::new(Self {
            data: AtomicPtr::new(ptr::null_mut()),
            file_index,
            is_level_0,
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

impl ProbLazyItem<ProbNode> {
    pub fn try_get_data<'a>(&self, cache: &HNSWIndexCache) -> Result<&'a ProbNode, BufIoError> {
        unsafe {
            if let Some(data) = self.data.load(Ordering::Relaxed).as_ref() {
                return Ok(data);
            }
            (*(cache.get_object(self.file_index, self.is_level_0)?)).try_get_data(cache)
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn get_absolute_latest_version<'a>(
        this: *mut Self,
        cache: &HNSWIndexCache,
    ) -> Result<(*mut Self, RwLockReadGuard<'a, (*mut Self, bool)>), BufIoError> {
        let self_ = unsafe { &*this };
        let guard = self_.try_get_data(cache)?.root_version.read();
        let (root, is_root) = *guard;
        if root.is_null() {
            return Ok((this, guard));
        }
        if !is_root {
            return Ok((root, guard));
        }
        drop(guard);
        let root_ref = unsafe { &*root };
        let guard = root_ref.try_get_data(cache)?.root_version.read();

        Ok((guard.0, guard))
    }

    #[allow(clippy::type_complexity)]
    pub fn get_absolute_latest_version_write_access<'a>(
        this: *mut Self,
        cache: &HNSWIndexCache,
    ) -> Result<(*mut Self, RwLockWriteGuard<'a, (*mut Self, bool)>), BufIoError> {
        let self_ = unsafe { &*this };
        let guard = self_.try_get_data(cache)?.root_version.write();
        let (root, is_root) = *guard;
        if root.is_null() {
            return Ok((this, guard));
        }
        if !is_root {
            return Ok((root, guard));
        }
        drop(guard);
        let root_ref = unsafe { &*root };
        let guard = root_ref.try_get_data(cache)?.root_version.write();

        Ok((guard.0, guard))
    }

    pub fn get_root_version(
        this: *mut Self,
        cache: &HNSWIndexCache,
    ) -> Result<*mut Self, BufIoError> {
        let self_ = unsafe { &*this };
        let (root, is_root) = *self_.try_get_data(cache)?.root_version.read();
        Ok(if root.is_null() || !is_root {
            this
        } else {
            root
        })
    }
}

impl ProbLazyItem<InvertedIndexNodeData> {
    pub fn try_get_data<'a>(
        &self,
        cache: &InvertedIndexCache,
        dim: u32,
    ) -> Result<&'a InvertedIndexNodeData, BufIoError> {
        unsafe {
            if let Some(data) = self.data.load(Ordering::Relaxed).as_ref() {
                return Ok(data);
            }

            let offset = self.file_index.offset;
            (*(cache.get_data(offset, (dim % cache.data_file_parts as u32) as u8)?))
                .try_get_data(cache, dim)
        }
    }
}

impl ProbLazyItem<TFIDFIndexNodeData> {
    pub fn try_get_data<'a>(
        &self,
        cache: &TFIDFIndexCache,
        dim: u32,
    ) -> Result<&'a TFIDFIndexNodeData, BufIoError> {
        unsafe {
            if let Some(data) = self.data.load(Ordering::Relaxed).as_ref() {
                return Ok(data);
            }

            let offset = self.file_index.offset;
            (*(cache.get_data(offset, (dim % cache.data_file_parts as u32) as u8)?))
                .try_get_data(cache, dim)
        }
    }
}

impl<T> Drop for ProbLazyItem<T> {
    fn drop(&mut self) {
        let data_ptr = self.data.load(Ordering::SeqCst);
        unsafe {
            if !data_ptr.is_null() {
                drop(Box::from_raw(data_ptr));
            }
        }
    }
}
