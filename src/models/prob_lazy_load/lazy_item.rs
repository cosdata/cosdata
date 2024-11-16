use std::{
    cell::Cell,
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering},
        Arc,
    },
};

use crate::models::{
    buffered_io::BufIoError,
    cache_loader::{Allocate, Allocator, ProbCache, ProbCacheable},
    lazy_load::{largest_power_of_4_below, FileIndex, SyncPersist},
    serializer::prob::{ProbSerialize, UpdateSerialized},
    types::FileOffset,
    versioning::Hash,
};

use super::lazy_item_array::ProbLazyItemArray;

// not cloneable, the data inside should NEVER be stored anywhere else, its wrapped in `Arc` to
// prevent use-after-free, as some methods may have exposed the data, while the state has been
// dropped
pub enum ProbLazyItemState<T> {
    Ready {
        data: Arc<T>,
        file_offset: Cell<Option<FileOffset>>,
        decay_counter: usize,
        persist_flag: AtomicBool,
        serialized_flag: AtomicBool,
        version_id: Hash,
        version_number: u16,
        versions: ProbLazyItemArray<T, 4>,
    },
    Pending {
        file_index: FileIndex,
    },
}

impl<T> ProbLazyItemState<T> {
    pub fn get_version_number(&self) -> u16 {
        match self {
            Self::Pending { file_index } => file_index.get_version_number().unwrap(),
            Self::Ready { version_number, .. } => *version_number,
        }
    }

    pub fn get_version_id(&self) -> Hash {
        match self {
            Self::Pending { file_index } => file_index.get_version_id().unwrap(),
            Self::Ready { version_id, .. } => *version_id,
        }
    }
}

pub struct ProbLazyItem<T> {
    state: Arc<AtomicPtr<ProbLazyItemState<T>>>,
    thread_lock: AtomicU32,
}

impl<T: Allocate> ProbLazyItem<T> {
    pub fn new(allocator: &Allocator, data: T, version_id: Hash, version_number: u16) -> *mut Self {
        allocator.alloc_item(Self {
            state: Arc::new(AtomicPtr::new(allocator.alloc_state(
                ProbLazyItemState::Ready {
                    data: Arc::new(data),
                    file_offset: Cell::new(None),
                    decay_counter: 0,
                    persist_flag: AtomicBool::new(true),
                    serialized_flag: AtomicBool::new(false),
                    version_id,
                    version_number,
                    versions: ProbLazyItemArray::new(),
                },
            ))),
            thread_lock: AtomicU32::new(u32::MAX),
        })
    }

    // the arc provided to this function should NOT be stored anywhere else, creating two LazyItem
    // with same value can cause undefined behavior, use `ProbLazyItem::new` if possible
    pub fn from_arc(
        allocator: &Allocator,
        data: Arc<T>,
        version_id: Hash,
        version_number: u16,
    ) -> *mut Self {
        allocator.alloc_item(Self {
            state: Arc::new(AtomicPtr::new(allocator.alloc_state(
                ProbLazyItemState::Ready {
                    data,
                    file_offset: Cell::new(None),
                    decay_counter: 0,
                    persist_flag: AtomicBool::new(true),
                    serialized_flag: AtomicBool::new(false),
                    version_id,
                    version_number,
                    versions: ProbLazyItemArray::new(),
                },
            ))),
            thread_lock: AtomicU32::new(u32::MAX),
        })
    }

    pub fn new_pending(allocator: &Allocator, file_index: FileIndex) -> *mut Self {
        allocator.alloc_item(Self {
            state: Arc::new(AtomicPtr::new(
                allocator.alloc_state(ProbLazyItemState::Pending { file_index }),
            )),
            thread_lock: AtomicU32::new(u32::MAX),
        })
    }
}

impl<T> ProbLazyItem<T> {
    pub fn lock(&self, thread_id: u32) {
        if self.thread_lock.load(Ordering::SeqCst) == thread_id {
            return;
        }
        #[allow(deprecated)]
        while self
            .thread_lock
            .compare_and_swap(u32::MAX, thread_id, Ordering::Acquire)
            == thread_id
        {}
    }

    pub fn release_lock(&self, thread_id: u32) {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(self.thread_lock.load(Ordering::SeqCst), thread_id);
        }

        self.thread_lock.store(u32::MAX, Ordering::Release);
    }

    pub fn swap_state(&self, new_state: *mut ProbLazyItemState<T>) -> *mut ProbLazyItemState<T> {
        self.state.swap(new_state, Ordering::SeqCst)
    }

    pub fn is_ready(&self) -> bool {
        unsafe {
            matches!(
                &*self.state.load(Ordering::SeqCst),
                ProbLazyItemState::Ready { .. }
            )
        }
    }

    pub fn is_pending(&self) -> bool {
        unsafe {
            matches!(
                &*self.state.load(Ordering::SeqCst),
                ProbLazyItemState::Pending { .. }
            )
        }
    }

    pub fn get_lazy_data(&self) -> Option<Arc<T>> {
        unsafe {
            let state = self.state.load(Ordering::SeqCst);
            match &*state {
                ProbLazyItemState::Pending { .. } => None,
                ProbLazyItemState::Ready { data, .. } => Some(data.clone()),
            }
        }
    }

    pub fn get_file_index(&self) -> Option<FileIndex> {
        unsafe {
            let state = self.state.load(Ordering::SeqCst);
            match &*state {
                ProbLazyItemState::Pending { file_index } => Some(file_index.clone()),
                ProbLazyItemState::Ready {
                    file_offset,
                    version_id,
                    version_number,
                    ..
                } => file_offset.get().map(|offset| FileIndex::Valid {
                    offset,
                    version_number: *version_number,
                    version_id: *version_id,
                }),
            }
        }
    }

    pub fn set_file_offset(&self, new_file_offset: FileOffset) {
        unsafe {
            let state = self.state.load(Ordering::SeqCst);
            if let ProbLazyItemState::Ready { file_offset, .. } = &*state {
                file_offset.set(Some(new_file_offset));
            }
        }
    }

    pub fn get_state(&self) -> Arc<AtomicPtr<ProbLazyItemState<T>>> {
        self.state.clone()
    }
}

impl<T: ProbCacheable + UpdateSerialized + ProbSerialize + Allocate> ProbLazyItem<T> {
    pub fn try_get_data(&self, cache: Arc<ProbCache>) -> Result<Arc<T>, BufIoError> {
        unsafe {
            match &*self.state.load(Ordering::SeqCst) {
                ProbLazyItemState::Ready { data, .. } => Ok(data.clone()),
                ProbLazyItemState::Pending { file_index } => Ok(cache
                    .get_object(file_index.clone(), 1000, &mut HashSet::new())?
                    .0),
            }
        }
    }

    pub fn try_get_versions(
        &self,
        cache: Arc<ProbCache>,
    ) -> Result<ProbLazyItemArray<T, 4>, BufIoError> {
        unsafe {
            match &*self.state.load(Ordering::SeqCst) {
                ProbLazyItemState::Ready { versions, .. } => Ok(versions.clone()),
                ProbLazyItemState::Pending { file_index } => Ok(cache
                    .get_object(file_index.clone(), 1000, &mut HashSet::new())?
                    .1),
            }
        }
    }

    pub fn add_version(
        this: *mut Self,
        version: *mut Self,
        cache: Arc<ProbCache>,
    ) -> Result<(), BufIoError> {
        let (_, latest_local_version_number) = Self::get_latest_version(this, cache.clone())?;
        let versions = unsafe { (*this).try_get_versions(cache)? };
        unsafe {
            (&*this).add_version_inner(version, 0, latest_local_version_number + 1, &versions);
        }
        Ok(())
    }

    pub fn add_version_inner(
        &self,
        version: *mut Self,
        self_relative_version_number: u16,
        target_relative_version_number: u16,
        versions: &ProbLazyItemArray<T, 4>,
    ) {
        let target_diff = target_relative_version_number - self_relative_version_number;
        let index = largest_power_of_4_below(target_diff);

        if let Some(existing_version) = versions.get(index as usize) {
            unsafe {
                (&*existing_version).add_version_inner(
                    version,
                    self_relative_version_number + (1 << (2 * index)),
                    target_relative_version_number,
                    versions,
                );
            }
        } else {
            debug_assert_eq!(versions.len(), index as usize);
            versions.push(version);
        }
    }

    pub fn get_latest_version(
        this: *mut Self,
        cache: Arc<ProbCache>,
    ) -> Result<(*mut Self, u16), BufIoError> {
        let versions = unsafe { (&*this).try_get_versions(cache.clone())? };
        if let Some(last) = versions.last() {
            let (latest_version, relative_local_version_number) =
                Self::get_latest_version(last, cache)?;
            Ok((
                latest_version,
                (1u16 << ((versions.len() as u8 - 1) * 2)) + relative_local_version_number,
            ))
        } else {
            Ok((this, 0))
        }
    }

    pub fn get_version(
        this: *mut Self,
        version: u16,
        cache: Arc<ProbCache>,
    ) -> Result<Option<*mut Self>, BufIoError> {
        let (version_number, versions) = unsafe {
            let this = &*this;
            (
                this.get_current_version_number(),
                this.try_get_versions(cache.clone())?,
            )
        };

        if version < version_number {
            return Ok(None);
        }

        if version == version_number {
            return Ok(Some(this));
        }

        let Some(mut prev) = versions.get(0) else {
            return Ok(None);
        };
        let mut i = 1;
        while let Some(next) = versions.get(i) {
            unsafe {
                if version < (&*next).get_current_version_number() {
                    return Self::get_version(prev, version, cache.clone());
                }
            }
            prev = next;
            i += 1;
        }

        Self::get_version(prev, version, cache)
    }
}

impl<T> SyncPersist for ProbLazyItem<T> {
    fn set_persistence(&self, flag: bool) {
        unsafe {
            if let ProbLazyItemState::Ready { persist_flag, .. } =
                &*self.state.load(Ordering::SeqCst)
            {
                persist_flag.store(flag, Ordering::SeqCst);
            }
        }
    }

    fn needs_persistence(&self) -> bool {
        unsafe {
            if let ProbLazyItemState::Ready { persist_flag, .. } =
                &*self.state.load(Ordering::SeqCst)
            {
                persist_flag.load(Ordering::SeqCst)
            } else {
                false
            }
        }
    }

    fn get_current_version(&self) -> Hash {
        unsafe { (*self.state.load(Ordering::SeqCst)).get_version_id() }
    }

    fn get_current_version_number(&self) -> u16 {
        unsafe { (*self.state.load(Ordering::SeqCst)).get_version_number() }
    }
}
