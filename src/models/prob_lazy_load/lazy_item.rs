use std::{
    cell::Cell,
    ops::Deref,
    ptr::NonNull,
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

// not cloneable
pub enum ProbLazyItemState<T> {
    Ready {
        data: T,
        file_offset: Cell<Option<FileOffset>>,
        persist_flag: AtomicBool,
        version_id: Hash,
        version_number: u16,
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

// The actual "lazy item", not used directly, instead a pointer (`*mut ProbLazyItemInner<T>`) or
// wrapper (`ProbLazyItem<T>`) is passed around and stored.
pub struct ProbLazyItemInner<T> {
    state: AtomicPtr<ProbLazyItemState<T>>,
}

// Just a convinient wrapper for `*mut ProbLazyItemInner<T>`
pub struct ProbLazyItem<T> {
    inner: NonNull<ProbLazyItemInner<T>>,
}

impl<T> Clone for ProbLazyItem<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self { inner: self.inner }
    }
}

impl<T> ProbLazyItem<T> {
    #[inline(always)]
    pub fn from_inner(inner: ProbLazyItemInner<T>) -> Self {
        let ptr = Box::into_raw(Box::new(inner));
        Self {
            inner: unsafe { NonNull::new_unchecked(ptr) },
        }
    }

    pub fn new(data: T, version_id: Hash, version_number: u16) -> Self {
        Self::from_inner(ProbLazyItemInner {
            state: AtomicPtr::new(Box::into_raw(Box::new(ProbLazyItemState::Ready {
                data,
                file_offset: Cell::new(None),
                persist_flag: AtomicBool::new(true),
                version_id,
                version_number,
            }))),
        })
    }

    pub fn new_from_state(state: ProbLazyItemState<T>) -> Self {
        Self::from_inner(ProbLazyItemInner {
            state: AtomicPtr::new(Box::into_raw(Box::new(state))),
        })
    }

    // the arc provided to this function should NOT be stored anywhere else, creating two LazyItem
    // with same value can cause undefined behavior, use `ProbLazyItem::new` if possible
    // pub fn from_arc(data: Arc<T>, version_id: Hash, version_number: u16) -> Self {
    //     Self::from_inner(ProbLazyItemInner {
    //         state: AtomicPtr::new(Box::into_raw(Box::new(ProbLazyItemState::Ready {
    //             data,
    //             file_offset: Cell::new(None),
    //             persist_flag: AtomicBool::new(true),
    //             version_id,
    //             version_number,
    //         }))),
    //     })
    // }

    pub fn new_pending(file_index: FileIndex) -> Self {
        Self::from_inner(ProbLazyItemInner {
            state: AtomicPtr::new(Box::into_raw(Box::new(ProbLazyItemState::Pending {
                file_index,
            }))),
        })
    }

    pub fn from_ptr(inner: *mut ProbLazyItemInner<T>) -> Self {
        Self {
            inner: unsafe { NonNull::new_unchecked(inner) },
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *mut ProbLazyItemInner<T> {
        self.inner.as_ptr()
    }
}

impl<T> Deref for ProbLazyItem<T> {
    type Target = ProbLazyItemInner<T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { self.inner.as_ref() }
    }
}

impl<T> ProbLazyItemInner<T> {
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
                ProbLazyItemState::Ready { .. }
            )
        }
    }

    pub fn is_pending(&self) -> bool {
        unsafe {
            matches!(
                &*self.state.load(Ordering::Acquire),
                ProbLazyItemState::Pending { .. }
            )
        }
    }

    pub fn get_lazy_data<'a>(&self) -> Option<&'a T> {
        unsafe {
            match &*self.state.load(Ordering::Acquire) {
                ProbLazyItemState::Pending { .. } => None,
                ProbLazyItemState::Ready { data, .. } => Some(&data),
            }
        }
    }

    pub fn get_file_index(&self) -> Option<FileIndex> {
        unsafe {
            match &*self.state.load(Ordering::Acquire) {
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
            if let ProbLazyItemState::Ready { file_offset, .. } =
                &*self.state.load(Ordering::Acquire)
            {
                file_offset.set(Some(new_file_offset));
            }
        }
    }
}

impl ProbLazyItem<ProbNode> {
    pub fn try_get_data<'a>(&self, cache: &ProbCache) -> Result<&'a ProbNode, BufIoError> {
        unsafe {
            match &*self.state.load(Ordering::Acquire) {
                ProbLazyItemState::Ready { data, .. } => Ok(data),
                ProbLazyItemState::Pending { file_index } => {
                    cache.get_object(file_index.clone())?.try_get_data(cache)
                }
            }
        }
    }

    pub fn add_version(
        &self,
        version: Self,
        cache: &ProbCache,
    ) -> Result<Result<Self, Self>, BufIoError> {
        let data = self.try_get_data(cache)?;
        let versions = &data.versions;

        let (_, latest_local_version_number) = self.get_latest_version_inner(versions, cache)?;

        let result = self.add_version_inner(version, 0, latest_local_version_number + 1, cache)?;

        Ok(result)
    }

    pub fn add_version_inner(
        &self,
        version: Self,
        self_relative_version_number: u16,
        target_relative_version_number: u16,
        cache: &ProbCache,
    ) -> Result<Result<Self, Self>, BufIoError> {
        let target_diff = target_relative_version_number - self_relative_version_number;
        if target_diff == 0 {
            return Ok(Err(self.clone()));
        }
        let index = largest_power_of_4_below(target_diff);
        let data = self.try_get_data(cache)?;
        let versions = &data.versions;

        if let Some(existing_version) = versions.get(index as usize) {
            return existing_version.add_version_inner(
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

    pub fn get_latest_version(&self, cache: &ProbCache) -> Result<(Self, u16), BufIoError> {
        let data = self.try_get_data(cache)?;
        let versions = &data.versions;

        self.get_latest_version_inner(versions, cache)
    }

    fn get_latest_version_inner(
        &self,
        versions: &ProbLazyItemArray<ProbNode, 4>,
        cache: &ProbCache,
    ) -> Result<(Self, u16), BufIoError> {
        if let Some(last) = versions.last() {
            let (latest_version, relative_local_version_number) = last.get_latest_version(cache)?;
            Ok((
                latest_version,
                (1u16 << ((versions.len() as u8 - 1) * 2)) + relative_local_version_number,
            ))
        } else {
            Ok((self.clone(), 0))
        }
    }

    pub fn get_version(&self, version: u16, cache: &ProbCache) -> Result<Option<Self>, BufIoError> {
        let version_number = self.get_current_version_number();
        let data = self.try_get_data(cache)?;
        let versions = &data.versions;

        if version < version_number {
            return Ok(None);
        }

        if version == version_number {
            return Ok(Some(self.clone()));
        }

        let Some(mut prev) = versions.get(0) else {
            return Ok(None);
        };
        let mut i = 1;
        while let Some(next) = versions.get(i) {
            if version < next.get_current_version_number() {
                return prev.get_version(version, cache);
            }
            prev = next;
            i += 1;
        }

        prev.get_version(version, cache)
    }
}

impl<T> SyncPersist for ProbLazyItemInner<T> {
    fn set_persistence(&self, flag: bool) {
        unsafe {
            if let ProbLazyItemState::Ready { persist_flag, .. } =
                &*self.state.load(Ordering::Acquire)
            {
                persist_flag.store(flag, Ordering::SeqCst);
            }
        }
    }

    fn needs_persistence(&self) -> bool {
        unsafe {
            if let ProbLazyItemState::Ready { persist_flag, .. } =
                &*self.state.load(Ordering::Acquire)
            {
                persist_flag.load(Ordering::SeqCst)
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

impl<T> std::hash::Hash for ProbLazyItem<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<T> PartialEq for ProbLazyItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }

    fn ne(&self, other: &Self) -> bool {
        (&self.state as *const AtomicPtr<_>).ne(&(&other.state as *const AtomicPtr<_>))
    }
}

impl<T> Eq for ProbLazyItem<T> {}

impl<T> Drop for ProbLazyItemInner<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: state must be a valid pointer
            drop(Box::from_raw(self.state.load(Ordering::SeqCst)));
        }
    }
}
