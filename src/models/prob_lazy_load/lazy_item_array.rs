use std::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use super::lazy_item::{ProbLazyItem, ProbLazyItemInner};

pub struct ProbLazyItemArray<T, const N: usize> {
    items: [AtomicPtr<ProbLazyItemInner<T>>; N],
}

impl<T, const N: usize> ProbLazyItemArray<T, N> {
    pub fn new() -> Self {
        Self {
            items: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }

    pub fn push(&self, item: ProbLazyItem<T>) {
        for i in 0..N {
            if self.items[i].load(Ordering::SeqCst).is_null() {
                self.items[i].store(item.as_ptr(), Ordering::SeqCst);
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

    pub fn last(&self) -> Option<ProbLazyItem<T>> {
        for i in (0..N).rev() {
            let ptr = self.items[i].load(Ordering::SeqCst);
            if !ptr.is_null() {
                return Some(ProbLazyItem::from_ptr(ptr));
            }
        }
        None
    }

    pub fn get(&self, idx: usize) -> Option<ProbLazyItem<T>> {
        if idx >= N || self.items[idx].load(Ordering::SeqCst).is_null() {
            return None;
        }
        Some(ProbLazyItem::from_ptr(
            self.items[idx].load(Ordering::SeqCst),
        ))
    }

    pub fn is_empty(&self) -> bool {
        self.items[0].load(Ordering::SeqCst).is_null()
    }
}
