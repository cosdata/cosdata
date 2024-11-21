use std::{
    ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};

use super::lazy_item::ProbLazyItem;

pub struct ProbLazyItemArray<T, const N: usize> {
    items: [AtomicPtr<Arc<ProbLazyItem<T>>>; N],
}

impl<T, const N: usize> ProbLazyItemArray<T, N> {
    pub fn new() -> Self {
        Self {
            items: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }

    pub fn push(&self, item: Arc<ProbLazyItem<T>>) {
        for i in 0..N {
            if self.items[i].load(Ordering::SeqCst).is_null() {
                self.items[i].store(Box::into_raw(Box::new(item)), Ordering::SeqCst);
                return;
            }
        }
        debug_assert!(false, "Array is full");
    }

    pub fn len(&self) -> usize {
        for i in 0..N {
            if self.items[i].load(Ordering::SeqCst).is_null() {
                return i;
            }
        }
        N
    }

    pub fn last(&self) -> Option<Arc<ProbLazyItem<T>>> {
        for i in (0..N).rev() {
            let ptr = self.items[i].load(Ordering::SeqCst);
            if !ptr.is_null() {
                return Some(unsafe { &*ptr }.clone());
            }
        }
        None
    }

    pub fn get(&self, idx: usize) -> Option<Arc<ProbLazyItem<T>>> {
        if idx >= N || self.items[idx].load(Ordering::SeqCst).is_null() {
            return None;
        }
        Some(unsafe { &*self.items[idx].load(Ordering::SeqCst) }.clone())
    }

    pub fn is_empty(&self) -> bool {
        self.items[0].load(Ordering::SeqCst).is_null()
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
