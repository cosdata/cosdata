use std::{
    fmt::Debug,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::{
    models::{
        buffered_io::BufIoError,
        cache_loader::{DenseIndexCache, InvertedIndexCache},
        fixedset::VersionedInvertedFixedSetIndex,
        lazy_load::{largest_power_of_4_below, FileIndex},
        prob_node::ProbNode,
        types::FileOffset,
        versioning::Hash,
    },
    storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasicTSHashmapData,
};

use super::lazy_item_array::ProbLazyItemArray;

#[derive(PartialEq, Debug)]
pub struct ReadyState<T> {
    pub data: T,
    pub file_offset: FileOffset,
    pub version_id: Hash,
    pub version_number: u16,
}

// not cloneable
#[derive(PartialEq, Debug)]
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
    pub is_level_0: bool,
}

impl<T: PartialEq> PartialEq for ProbLazyItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.is_level_0 == other.is_level_0
            && unsafe {
                *self.state.load(Ordering::Relaxed) == *other.state.load(Ordering::Relaxed)
            }
    }
}

impl<T: Debug> Debug for ProbLazyItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProbLazyItem")
            .field("state", unsafe { &*self.state.load(Ordering::Relaxed) })
            .field("is_level_0", &self.is_level_0)
            .finish()
    }
}

impl<T> ProbLazyItem<T> {
    pub fn new(
        data: T,
        version_id: Hash,
        version_number: u16,
        is_level_0: bool,
        file_offset: FileOffset,
    ) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(ProbLazyItemState::Ready(
                ReadyState {
                    data,
                    file_offset,
                    version_id,
                    version_number,
                },
            )))),
            is_level_0,
        }))
    }

    pub fn new_from_state(state: ProbLazyItemState<T>, is_level_0: bool) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(state))),
            is_level_0,
        }))
    }

    pub fn new_pending(file_index: FileIndex, is_level_0: bool) -> *mut Self {
        Box::into_raw(Box::new(Self {
            state: AtomicPtr::new(Box::into_raw(Box::new(ProbLazyItemState::Pending(
                file_index,
            )))),
            is_level_0,
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

    pub fn get_file_index(&self) -> FileIndex {
        unsafe {
            match &*self.state.load(Ordering::Acquire) {
                ProbLazyItemState::Pending(file_index) => file_index.clone(),
                ProbLazyItemState::Ready(state) => FileIndex::Valid {
                    offset: state.file_offset,
                    version_number: state.version_number,
                    version_id: state.version_id,
                },
            }
        }
    }

    pub fn get_current_version_id(&self) -> Hash {
        unsafe { (*self.state.load(Ordering::Acquire)).get_version_id() }
    }

    pub fn get_current_version_number(&self) -> u16 {
        unsafe { (*self.state.load(Ordering::Acquire)).get_version_number() }
    }
}

impl ProbLazyItem<ProbNode> {
    pub fn try_get_data<'a>(&self, cache: &DenseIndexCache) -> Result<&'a ProbNode, BufIoError> {
        unsafe {
            match &*self.state.load(Ordering::Relaxed) {
                ProbLazyItemState::Ready(state) => Ok(&state.data),
                ProbLazyItemState::Pending(file_index) => {
                    (*(cache.get_object(file_index.clone(), self.is_level_0)?)).try_get_data(cache)
                }
            }
        }
    }

    pub fn add_version(
        this: *mut Self,
        version: *mut Self,
        cache: &DenseIndexCache,
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
        cache: &DenseIndexCache,
    ) -> Result<Result<*mut Self, *mut Self>, BufIoError> {
        let target_diff = target_relative_version_number - self_relative_version_number;
        if target_diff == 0 {
            return Ok(Err(this));
        }
        let index = largest_power_of_4_below(target_diff);
        let data = unsafe { &*this }.try_get_data(cache)?;
        let versions = &data.versions;

        if let Some(existing_version) = versions.get(index as usize) {
            Self::add_version_inner(
                existing_version,
                version,
                self_relative_version_number + (1 << (2 * index)),
                target_relative_version_number,
                cache,
            )
        } else {
            debug_assert_eq!(versions.len(), index as usize);
            versions.push(version.clone());
            Ok(Ok(this))
        }
    }

    pub fn get_latest_version(
        this: *mut Self,
        cache: &DenseIndexCache,
    ) -> Result<(*mut Self, u16), BufIoError> {
        let data = unsafe { &*this }.try_get_data(cache)?;
        let versions = &data.versions;

        Self::get_latest_version_inner(this, versions, cache)
    }

    fn get_latest_version_inner<const LEN: usize>(
        this: *mut Self,
        versions: &ProbLazyItemArray<ProbNode, LEN>,
        cache: &DenseIndexCache,
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

    pub fn get_root_version(
        this: *mut Self,
        cache: &DenseIndexCache,
    ) -> Result<*mut Self, BufIoError> {
        let self_ = unsafe { &*this };
        let root = self_.try_get_data(cache)?.root_version;
        Ok(if root.is_null() { this } else { root })
    }

    pub fn get_version(
        this: *mut Self,
        version: u16,
        cache: &DenseIndexCache,
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

impl ProbLazyItem<InvertedIndexSparseAnnNodeBasicTSHashmapData> {
    pub fn try_get_data<'a>(
        &self,
        cache: &InvertedIndexCache,
        dim: u32,
    ) -> Result<&'a InvertedIndexSparseAnnNodeBasicTSHashmapData, BufIoError> {
        unsafe {
            match &*self.state.load(Ordering::Relaxed) {
                ProbLazyItemState::Ready(state) => Ok(&state.data),
                ProbLazyItemState::Pending(file_index) => {
                    let offset = file_index.get_offset().unwrap();
                    (*(cache.get_data(offset, (dim % cache.data_file_parts as u32) as u8)?))
                        .try_get_data(cache, dim)
                }
            }
        }
    }
}

impl ProbLazyItem<VersionedInvertedFixedSetIndex> {
    pub fn try_get_data<'a>(
        &self,
        cache: &InvertedIndexCache,
        dim: u32,
    ) -> Result<&'a VersionedInvertedFixedSetIndex, BufIoError> {
        unsafe {
            match &*self.state.load(Ordering::Relaxed) {
                ProbLazyItemState::Ready(state) => Ok(&state.data),
                ProbLazyItemState::Pending(file_index) => {
                    let offset = file_index.get_offset().unwrap();
                    (*(cache.get_sets(offset, (dim % cache.data_file_parts as u32) as u8)?))
                        .try_get_data(cache, dim)
                }
            }
        }
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
