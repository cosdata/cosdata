use std::{
    cell::Cell,
    sync::atomic::{AtomicBool, AtomicPtr, Ordering},
};

use crate::models::{
    buffered_io::BufIoError,
    cache_loader::ProbCache,
    lazy_load::{largest_power_of_4_below, FileIndex, SyncPersist},
    prob_node::ProbNode,
    types::FileOffset,
    versioning::Hash,
};

use super::lazy_item_array::ProbLazyItemArray;

pub struct ReadyState<T> {
    pub data: T,
    pub file_offset: Cell<Option<FileOffset>>,
    pub persist_flag: AtomicBool,
    pub version_id: Hash,
    pub version_number: u16,
}

// not cloneable
pub enum ProbLazyItemState<T> {
    Ready(ReadyState<T>),
    Pending(FileIndex),
}

impl<T> ProbLazyItemState<T> {
    pub fn get_version_number(&self) -> u16 {
        match self {
            Self::Pending(file_index) => file_index.get_version_number().unwrap(),
            Self::Ready(state) => state.version_number,
        }
    }

    pub fn get_version_id(&self) -> Hash {
        match self {
            Self::Pending(file_index) => file_index.get_version_id().unwrap(),
            Self::Ready(state) => state.version_id,
        }
    }
}

pub struct ProbLazyItem<T> {
    state: AtomicPtr<ProbLazyItemState<T>>,
}

impl<T> ProbLazyItem<T> {
    pub fn new(data: T, version_id: Hash, version_number: u16) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(ProbLazyItemState::Ready(
                ReadyState {
                    data,
                    file_offset: Cell::new(None),
                    persist_flag: AtomicBool::new(true),
                    version_id,
                    version_number,
                },
            )))),
        }))
    }

    pub fn new_from_state(state: ProbLazyItemState<T>) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(state))),
        }))
    }

    pub fn new_pending(file_index: FileIndex) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(ProbLazyItemState::Pending(
                file_index,
            )))),
        }))
    }

    pub fn unsafe_get_state(&self) -> &ProbLazyItemState<T> {
        // SAFETY: caller must make sure the state is not dropped by some other thread
        unsafe { &*self.state.load(Ordering::Acquire) }
    }

    pub fn set_state(&self, new_state: ProbLazyItemState<T>) {
        let old_state = self
            .state
            .swap(Box::into_raw(Box::new(new_state)), Ordering::SeqCst);
        unsafe {
            // SAFETY: state must be a valid pointer
            drop(Box::from_raw(old_state));
        }
    }

    pub fn is_ready(&self) -> bool {
        unsafe {
            matches!(
                &*self.state.load(Ordering::Acquire),
                ProbLazyItemState::Ready(_)
            )
        }
    }

    pub fn is_pending(&self) -> bool {
        unsafe {
            matches!(
                &*self.state.load(Ordering::Acquire),
                ProbLazyItemState::Pending(_)
            )
        }
    }

    pub fn get_lazy_data<'a>(&self) -> Option<&'a T> {
        unsafe {
            match &*self.state.load(Ordering::Acquire) {
                ProbLazyItemState::Pending(_) => None,
                ProbLazyItemState::Ready(state) => Some(&state.data),
            }
        }
    }

    pub fn get_file_index(&self) -> Option<FileIndex> {
        unsafe {
            match &*self.state.load(Ordering::Acquire) {
                ProbLazyItemState::Pending(file_index) => Some(file_index.clone()),
                ProbLazyItemState::Ready(state) => {
                    state.file_offset.get().map(|offset| FileIndex::Valid {
                        offset,
                        version_number: state.version_number,
                        version_id: state.version_id,
                    })
                }
            }
        }
    }

    pub fn set_file_offset(&self, new_file_offset: FileOffset) {
        unsafe {
            if let ProbLazyItemState::Ready(state) = &*self.state.load(Ordering::Acquire) {
                state.file_offset.set(Some(new_file_offset));
            }
        }
    }
}

impl ProbLazyItem<ProbNode> {
    pub fn try_get_data<'a>(&self, cache: &ProbCache) -> Result<&'a ProbNode, BufIoError> {
        unsafe {
            match &*self.state.load(Ordering::Relaxed) {
                ProbLazyItemState::Ready(state) => Ok(&state.data),
                ProbLazyItemState::Pending(file_index) => {
                    (&*(cache.get_object(file_index.clone())?)).try_get_data(cache)
                }
            }
        }
    }

    pub fn add_version(
        this: *mut Self,
        version: *mut Self,
        cache: &ProbCache,
    ) -> Result<Result<*mut Self, *mut Self>, BufIoError> {
        let data = unsafe { &*this }.try_get_data(cache)?;
        let versions = &data.versions;

        let (_, latest_local_version_number) =
            Self::get_latest_version_inner(this, versions, cache)?;

        let result =
            Self::add_version_inner(this, version, 0, latest_local_version_number + 1, cache)?;

        Ok(result)
    }

    pub fn add_version_inner(
        this: *mut Self,
        version: *mut Self,
        self_relative_version_number: u16,
        target_relative_version_number: u16,
        cache: &ProbCache,
    ) -> Result<Result<*mut Self, *mut Self>, BufIoError> {
        let target_diff = target_relative_version_number - self_relative_version_number;
        if target_diff == 0 {
            return Ok(Err(this));
        }
        let index = largest_power_of_4_below(target_diff);
        let data = unsafe { &*this }.try_get_data(cache)?;
        let versions = &data.versions;

        if let Some(existing_version) = versions.get(index as usize) {
            return Self::add_version_inner(
                existing_version,
                version,
                self_relative_version_number + (1 << (2 * index)),
                target_relative_version_number,
                cache,
            );
        } else {
            debug_assert_eq!(versions.len(), index as usize);
            versions.push(version.clone());
        }

        Ok(Ok(version))
    }

    pub fn get_latest_version(
        this: *mut Self,
        cache: &ProbCache,
    ) -> Result<(*mut Self, u16), BufIoError> {
        let data = unsafe { &*this }.try_get_data(cache)?;
        let versions = &data.versions;

        Self::get_latest_version_inner(this, versions, cache)
    }

    fn get_latest_version_inner(
        this: *mut Self,
        versions: &ProbLazyItemArray<ProbNode, 4>,
        cache: &ProbCache,
    ) -> Result<(*mut Self, u16), BufIoError> {
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
        cache: &ProbCache,
    ) -> Result<Option<*mut Self>, BufIoError> {
        let self_ = unsafe { &*this };
        let version_number = self_.get_current_version_number();
        let data = self_.try_get_data(cache)?;
        let versions = &data.versions;

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
            if version < unsafe { &*next }.get_current_version_number() {
                return Self::get_version(prev, version, cache);
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
            if let ProbLazyItemState::Ready(state) = &*self.state.load(Ordering::Acquire) {
                state.persist_flag.store(flag, Ordering::SeqCst);
            }
        }
    }

    fn needs_persistence(&self) -> bool {
        unsafe {
            if let ProbLazyItemState::Ready(state) = &*self.state.load(Ordering::Acquire) {
                state.persist_flag.load(Ordering::SeqCst)
            } else {
                false
            }
        }
    }

    fn get_current_version(&self) -> Hash {
        unsafe { (*self.state.load(Ordering::Acquire)).get_version_id() }
    }

    fn get_current_version_number(&self) -> u16 {
        unsafe { (*self.state.load(Ordering::Acquire)).get_version_number() }
    }
}

impl<T> Drop for ProbLazyItem<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: state must be a valid pointer
            drop(Box::from_raw(self.state.load(Ordering::SeqCst)));
        }
    }
}
