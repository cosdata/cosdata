use std::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use super::lazy_item::ProbLazyItem;

pub struct ProbLazyItemArray<T, const N: usize> {
    items: [AtomicPtr<ProbLazyItem<T>>; N],
}

impl<T, const N: usize> ProbLazyItemArray<T, N> {
    pub fn new() -> Self {
        Self {
            items: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }

    pub fn push(&self, item: *mut ProbLazyItem<T>) {
        for i in 0..N {
            if self.items[i].load(Ordering::SeqCst).is_null() {
                self.items[i].store(item, Ordering::SeqCst);
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

    pub fn last(&self) -> Option<*mut ProbLazyItem<T>> {
        for i in (0..N).rev() {
            let ptr = self.items[i].load(Ordering::SeqCst);
            if !ptr.is_null() {
                return Some(ptr);
            }
        }
        None
    }

    pub fn get(&self, idx: usize) -> Option<*mut ProbLazyItem<T>> {
        if idx >= N || self.items[idx].load(Ordering::SeqCst).is_null() {
            return None;
        }
        Some(self.items[idx].load(Ordering::SeqCst))
    }

    pub fn is_empty(&self) -> bool {
        self.items[0].load(Ordering::SeqCst).is_null()
    }

    pub fn insert(&self, idx: usize, value: *mut ProbLazyItem<T>) {
        self.items[idx].store(value, Ordering::SeqCst);
    }

    pub fn get_or_insert<F>(&self, idx: usize, mut f: F) -> *mut ProbLazyItem<T>
    where
        F: FnMut() -> *mut ProbLazyItem<T>,
    {
        let mut return_value = ptr::null_mut();
        let _ = self.items[idx].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |existing| {
            if existing.is_null() {
                let value = f();
                return_value = value;
                Some(value)
            } else {
                return_value = existing;
                None
            }
        });
        return_value
    }
}
