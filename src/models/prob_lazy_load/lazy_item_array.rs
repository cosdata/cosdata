use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, AtomicUsize, Ordering},
        Arc,
    },
};

use super::lazy_item::ProbLazyItem;

pub struct ProbLazyItemArray<T, const N: usize> {
    items: [AtomicPtr<Arc<ProbLazyItem<T>>>; N],
    len: AtomicUsize,
}

impl<T, const N: usize> ProbLazyItemArray<T, N> {
    pub fn new() -> Self {
        Self {
            items: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
            len: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, item: Arc<ProbLazyItem<T>>) {
        let idx = self.len.fetch_add(1, Ordering::SeqCst);
        debug_assert!(idx < N);
        self.items[idx].store(Box::into_raw(Box::new(item)), Ordering::SeqCst);
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::SeqCst)
    }

    pub fn last(&self) -> Option<Arc<ProbLazyItem<T>>> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        Some(unsafe { &*self.items[len - 1].load(Ordering::SeqCst) }.clone())
    }

    pub fn get(&self, idx: usize) -> Option<Arc<ProbLazyItem<T>>> {
        let len = self.len();
        if idx >= len {
            return None;
        }
        Some(unsafe { &*self.items[idx].load(Ordering::SeqCst) }.clone())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize> Drop for ProbLazyItemArray<T, N> {
    fn drop(&mut self) {
        let len = self.len();

        for i in 0..len {
            unsafe {
                let ptr = self.items[i].load(Ordering::SeqCst);
                drop(Box::from_raw(ptr));
            }
        }
    }
}
