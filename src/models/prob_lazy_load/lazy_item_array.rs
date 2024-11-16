use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, AtomicUsize, Ordering},
        Arc,
    },
};

use super::lazy_item::ProbLazyItem;

pub struct ProbLazyItemArray<T, const N: usize> {
    items: Arc<[Arc<AtomicPtr<ProbLazyItem<T>>>; N]>,
    len: Arc<AtomicUsize>,
}

impl<T, const N: usize> Clone for ProbLazyItemArray<T, N> {
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
            len: self.len.clone(),
        }
    }
}

impl<T, const N: usize> ProbLazyItemArray<T, N> {
    pub fn new() -> Self {
        Self {
            items: Arc::new(std::array::from_fn(|_| {
                Arc::new(AtomicPtr::new(ptr::null_mut()))
            })),
            len: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn push(&self, item: *mut ProbLazyItem<T>) {
        let idx = self.len.fetch_add(1, Ordering::SeqCst);
        debug_assert!(idx < N);
        self.items[idx].store(item, Ordering::SeqCst);
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::SeqCst)
    }

    pub fn last(&self) -> Option<*mut ProbLazyItem<T>> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        Some(self.items[len - 1].load(Ordering::SeqCst))
    }

    pub fn get(&self, idx: usize) -> Option<*mut ProbLazyItem<T>> {
        let len = self.len();
        if idx >= len {
            return None;
        }
        Some(self.items[idx].load(Ordering::SeqCst))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
