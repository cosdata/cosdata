use std::{
    cell::Cell,
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, AtomicPtr, Ordering},
        Arc,
    },
};

use crate::models::{
    buffered_io::BufIoError,
    cache_loader::{ProbCache, ProbCacheable},
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
    state: AtomicPtr<Arc<ProbLazyItemState<T>>>,
}

impl<T> ProbLazyItem<T> {
    pub fn new(data: T, version_id: Hash, version_number: u16) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(Arc::new(
                ProbLazyItemState::Ready {
                    data: Arc::new(data),
                    file_offset: Cell::new(None),
                    decay_counter: 0,
                    persist_flag: AtomicBool::new(true),
                    version_id,
                    version_number,
                    versions: ProbLazyItemArray::new(),
                },
            )))),
        }))
    }

    // the arc provided to this function should NOT be stored anywhere else, creating two LazyItem
    // with same value can cause undefined behavior, use `ProbLazyItem::new` if possible
    pub fn from_arc(data: Arc<T>, version_id: Hash, version_number: u16) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(Arc::new(
                ProbLazyItemState::Ready {
                    data,
                    file_offset: Cell::new(None),
                    decay_counter: 0,
                    persist_flag: AtomicBool::new(true),
                    version_id,
                    version_number,
                    versions: ProbLazyItemArray::new(),
                },
            )))),
        }))
    }

    pub fn new_pending(file_index: FileIndex) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(Arc::new(
                ProbLazyItemState::Pending { file_index },
            )))),
        }))
    }

    pub fn get_state(&self) -> Arc<ProbLazyItemState<T>> {
        unsafe { (*self.state.load(Ordering::SeqCst)).clone() }
    }

    pub fn swap_state(
        &self,
        new_state: *mut Arc<ProbLazyItemState<T>>,
    ) -> *mut Arc<ProbLazyItemState<T>> {
        self.state.swap(new_state, Ordering::SeqCst)
    }

    pub fn is_ready(&self) -> bool {
        matches!(&*self.get_state(), ProbLazyItemState::Ready { .. })
    }

    pub fn is_pending(&self) -> bool {
        matches!(&*self.get_state(), ProbLazyItemState::Pending { .. })
    }

    pub fn get_lazy_data(&self) -> Option<Arc<T>> {
        match &*self.get_state() {
            ProbLazyItemState::Pending { .. } => None,
            ProbLazyItemState::Ready { data, .. } => Some(data.clone()),
        }
    }

    pub fn get_file_index(&self) -> Option<FileIndex> {
        match &*self.get_state() {
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

    pub fn set_file_offset(&self, new_file_offset: FileOffset) {
        if let ProbLazyItemState::Ready { file_offset, .. } = &*self.get_state() {
            file_offset.set(Some(new_file_offset));
        }
    }
}

impl<T: ProbCacheable + UpdateSerialized + ProbSerialize> ProbLazyItem<T> {
    pub fn try_get_data(&self, cache: Arc<ProbCache>) -> Result<Arc<T>, BufIoError> {
        match &*self.get_state() {
            ProbLazyItemState::Ready { data, .. } => Ok(data.clone()),
            ProbLazyItemState::Pending { file_index } => Ok(cache
                .get_object(file_index.clone(), 1000, &mut HashSet::new())?
                .0),
        }
    }

    pub fn try_get_versions(
        &self,
        cache: Arc<ProbCache>,
    ) -> Result<ProbLazyItemArray<T, 4>, BufIoError> {
        match &*self.get_state() {
            ProbLazyItemState::Ready { versions, .. } => Ok(versions.clone()),
            ProbLazyItemState::Pending { file_index } => {
                let (_, versions) =
                    cache.get_object(file_index.clone(), 1000, &mut HashSet::new())?;
                Ok(versions)
            }
        }
    }

    pub fn set_versions_persistence(
        &self,
        flag: bool,
        cache: Arc<ProbCache>,
    ) -> Result<(), BufIoError> {
        self.set_persistence(flag);
        let versions = self.try_get_versions(cache.clone())?;

        for i in 0..4 {
            let Some(version) = versions.get(i) else {
                break;
            };

            unsafe { &*version }.set_versions_persistence(flag, cache.clone())?;
        }

        Ok(())
    }

    pub fn add_version(
        this: *mut Self,
        version: *mut Self,
        cache: Arc<ProbCache>,
    ) -> Result<Result<*mut Self, *mut Self>, BufIoError> {
        let versions = unsafe { &*this }.try_get_versions(cache.clone())?;

        let (_, latest_local_version_number) =
            Self::get_latest_version_inner(this, versions.clone(), cache.clone())?;
        let result = ProbLazyItem::add_version_inner(
            this,
            version,
            0,
            latest_local_version_number + 1,
            cache.clone(),
        )?;

        Ok(result)
    }

    pub fn add_version_inner(
        this: *mut Self,
        version: *mut Self,
        self_relative_version_number: u16,
        target_relative_version_number: u16,
        cache: Arc<ProbCache>,
    ) -> Result<Result<*mut Self, *mut Self>, BufIoError> {
        let target_diff = target_relative_version_number - self_relative_version_number;
        if target_diff == 0 {
            return Ok(Err(this));
        }
        let index = largest_power_of_4_below(target_diff);
        let versions = unsafe { &*this }.try_get_versions(cache.clone())?;

        if let Some(existing_version) = versions.get(index as usize) {
            return ProbLazyItem::add_version_inner(
                existing_version,
                version,
                self_relative_version_number + (1 << (2 * index)),
                target_relative_version_number,
                cache.clone(),
            );
        } else {
            debug_assert_eq!(versions.len(), index as usize);
            versions.push(version);
        }

        Ok(Ok(version))
    }

    pub fn get_latest_version(
        this: *mut Self,
        cache: Arc<ProbCache>,
    ) -> Result<(*mut Self, u16), BufIoError> {
        let versions = unsafe { (&*this).try_get_versions(cache.clone())? };

        Self::get_latest_version_inner(this, versions, cache)
    }

    fn get_latest_version_inner(
        this: *mut Self,
        versions: ProbLazyItemArray<T, 4>,
        cache: Arc<ProbCache>,
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
        if let ProbLazyItemState::Ready { persist_flag, .. } = &*self.get_state() {
            persist_flag.store(flag, Ordering::SeqCst);
        }
    }

    fn needs_persistence(&self) -> bool {
        if let ProbLazyItemState::Ready { persist_flag, .. } = &*self.get_state() {
            persist_flag.load(Ordering::SeqCst)
        } else {
            false
        }
    }

    fn get_current_version(&self) -> Hash {
        unsafe {
            self.state
                .load(Ordering::SeqCst)
                .as_ref()
                .unwrap()
                .get_version_id()
        }
    }

    fn get_current_version_number(&self) -> u16 {
        unsafe {
            self.state
                .load(Ordering::SeqCst)
                .as_ref()
                .unwrap()
                .get_version_number()
        }
    }
}
