use std::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

pub struct AtomicArray<T, const N: usize> {
    items: [AtomicPtr<T>; N],
}

#[cfg(test)]
impl<T: PartialEq, const N: usize> PartialEq for AtomicArray<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.items.len() == other.items.len()
            && self.items.iter().zip(&other.items).all(|(s, o)| unsafe {
                s.load(Ordering::Relaxed).as_ref() == o.load(Ordering::Relaxed).as_ref()
            })
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug, const N: usize> std::fmt::Debug for AtomicArray<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.items
                    .iter()
                    .map(|i| unsafe { i.load(Ordering::Relaxed).as_ref() }),
            )
            .finish()
    }
}

impl<T, const N: usize> AtomicArray<T, N> {
    pub fn new() -> Self {
        Self {
            items: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }

    pub fn push(&self, item: *mut T) {
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

    pub fn last(&self) -> Option<*mut T> {
        for i in (0..N).rev() {
            let ptr = self.items[i].load(Ordering::SeqCst);
            if !ptr.is_null() {
                return Some(ptr);
            }
        }
        None
    }

    pub fn get(&self, idx: usize) -> Option<*mut T> {
        if idx >= N || self.items[idx].load(Ordering::SeqCst).is_null() {
            return None;
        }
        Some(self.items[idx].load(Ordering::SeqCst))
    }

    pub fn is_empty(&self) -> bool {
        self.items[0].load(Ordering::SeqCst).is_null()
    }

    pub fn insert(&self, idx: usize, value: *mut T) {
        self.items[idx].store(value, Ordering::SeqCst);
    }

    pub fn get_or_insert<F>(&self, idx: usize, mut f: F) -> *mut T
    where
        F: FnMut() -> *mut T,
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
