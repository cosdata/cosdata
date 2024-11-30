use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicBool, AtomicPtr, Ordering},
        Arc,
    },
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

// not cloneable, the data inside should NEVER be stored anywhere else, its wrapped in `Arc` to
// prevent use-after-free, as some methods may have exposed the data, while the state has been
// dropped
pub enum ProbLazyItemState<T: ?Sized> {
    Ready {
        data: Arc<T>,
        file_offset: Cell<Option<FileOffset>>,
        persist_flag: AtomicBool,
        version_id: Hash,
        version_number: u16,
    },
    Pending {
        file_index: FileIndex,
    },
}

impl<T: ?Sized> ProbLazyItemState<T> {
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

pub struct ProbLazyItem<T: ?Sized> {
    state: AtomicPtr<Arc<ProbLazyItemState<T>>>,
}

impl<T> ProbLazyItem<T> {
    pub fn new(data: T, version_id: Hash, version_number: u16) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(Arc::new(
                ProbLazyItemState::Ready {
                    data: Arc::new(data),
                    file_offset: Cell::new(None),
                    persist_flag: AtomicBool::new(true),
                    version_id,
                    version_number,
                },
            )))),
        })
    }

    pub fn new_from_state(state: ProbLazyItemState<T>) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(Arc::new(state)))),
        })
    }

    // the arc provided to this function should NOT be stored anywhere else, creating two LazyItem
    // with same value can cause undefined behavior, use `ProbLazyItem::new` if possible
    pub fn from_arc(data: Arc<T>, version_id: Hash, version_number: u16) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(Arc::new(
                ProbLazyItemState::Ready {
                    data,
                    file_offset: Cell::new(None),
                    persist_flag: AtomicBool::new(true),
                    version_id,
                    version_number,
                },
            )))),
        })
    }

    pub fn new_pending(file_index: FileIndex) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(Arc::new(
                ProbLazyItemState::Pending { file_index },
            )))),
        })
    }

    pub fn get_state(&self) -> Arc<ProbLazyItemState<T>> {
        unsafe {
            // SAFETY: state must be a valid pointer
            (*self.state.load(Ordering::SeqCst)).clone()
        }
    }

    pub fn set_state(&self, new_state: Arc<ProbLazyItemState<T>>) {
        let old_state = self
            .state
            .swap(Box::into_raw(Box::new(new_state)), Ordering::SeqCst);
        unsafe {
            // SAFETY: state must be a valid pointer
            drop(Box::from_raw(old_state));
        }
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

impl ProbLazyItem<ProbNode> {
    pub fn try_get_data(&self, cache: &Arc<ProbCache>) -> Result<Arc<ProbNode>, BufIoError> {
        unsafe {
            match &**self.state.load(Ordering::Acquire) {
                ProbLazyItemState::Ready { data, .. } => Ok(data.clone()),
                ProbLazyItemState::Pending { file_index } => {
                    Ok(cache.get_object(file_index.clone())?)
                }
            }
        }
    }

    pub fn add_version(
        self: &Arc<Self>,
        version: Arc<Self>,
        cache: &Arc<ProbCache>,
    ) -> Result<Result<Arc<Self>, Arc<Self>>, BufIoError> {
        let data = self.try_get_data(cache)?;
        let versions = &data.versions;

        let (_, latest_local_version_number) = self.get_latest_version_inner(versions, cache)?;

        let result = self.add_version_inner(version, 0, latest_local_version_number + 1, cache)?;

        Ok(result)
    }

    pub fn add_version_inner(
        self: &Arc<Self>,
        version: Arc<Self>,
        self_relative_version_number: u16,
        target_relative_version_number: u16,
        cache: &Arc<ProbCache>,
    ) -> Result<Result<Arc<Self>, Arc<Self>>, BufIoError> {
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

    pub fn get_latest_version(
        self: &Arc<Self>,
        cache: &Arc<ProbCache>,
    ) -> Result<(Arc<Self>, u16), BufIoError> {
        let data = self.try_get_data(cache)?;
        let versions = &data.versions;

        self.get_latest_version_inner(versions, cache)
    }

    fn get_latest_version_inner(
        self: &Arc<Self>,
        versions: &ProbLazyItemArray<ProbNode, 4>,
        cache: &Arc<ProbCache>,
    ) -> Result<(Arc<Self>, u16), BufIoError> {
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

    pub fn get_version(
        self: &Arc<Self>,
        version: u16,
        cache: &Arc<ProbCache>,
    ) -> Result<Option<Arc<Self>>, BufIoError> {
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
        self.get_state().get_version_id()
    }

    fn get_current_version_number(&self) -> u16 {
        self.get_state().get_version_number()
    }
}

impl<T> std::hash::Hash for ProbLazyItem<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (&self.state as *const AtomicPtr<_>).hash(state);
    }
}

impl<T> PartialEq for ProbLazyItem<T> {
    fn eq(&self, other: &Self) -> bool {
        (&self.state as *const AtomicPtr<_>).eq(&(&other.state as *const AtomicPtr<_>))
    }

    fn ne(&self, other: &Self) -> bool {
        (&self.state as *const AtomicPtr<_>).ne(&(&other.state as *const AtomicPtr<_>))
    }
}

impl<T> Eq for ProbLazyItem<T> {}

impl<T: ?Sized> Drop for ProbLazyItem<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: state must be a valid pointer
            drop(Box::from_raw(self.state.load(Ordering::SeqCst)));
        }
    }
}
