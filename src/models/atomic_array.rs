use std::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

pub struct AtomicArray<T, const N: usize> {
    pub(crate) items: [AtomicPtr<T>; N],
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

impl<T, const N: usize> Default for AtomicArray<T, N> {
    fn default() -> Self {
        Self {
            items: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }
}

#[allow(unused)]
impl<T, const N: usize> AtomicArray<T, N> {
    pub fn new() -> Self {
        Self::default()
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
        let mut n = 0;
        for i in 0..N {
            if !self.items[i].load(Ordering::SeqCst).is_null() {
                n += 1;
            }
        }
        n
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
        if idx >= N {
            return None;
        }
        let ptr = self.items[idx].load(Ordering::SeqCst);
        if ptr.is_null() {
            return None;
        }
        Some(ptr)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&self, idx: usize, value: *mut T) {
        self.items[idx].store(value, Ordering::SeqCst);
    }

    pub fn get_or_insert<F>(&self, idx: usize, mut f: F) -> (*mut T, bool)
    where
        F: FnMut() -> *mut T,
    {
        let mut return_value = ptr::null_mut();
        let res = self.items[idx].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |existing| {
            if existing.is_null() {
                let value = f();
                return_value = value;
                Some(value)
            } else {
                return_value = existing;
                None
            }
        });
        (return_value, res.is_ok())
    }

    pub fn get_or_insert_with<F>(&self, idx: usize, mut f: F) -> (*mut T, bool)
    where
        F: FnMut() -> T,
    {
        let mut return_value = ptr::null_mut();
        let mut v: *mut T = ptr::null_mut();
        let res = self.items[idx].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |existing| {
            if existing.is_null() {
                if v.is_null() {
                    v = Box::into_raw(Box::new(f()))
                }
                return_value = v;
                Some(v)
            } else {
                return_value = existing;
                None
            }
        });
        if !v.is_null() && res.is_err() {
            unsafe {
                drop(Box::from_raw(v));
            }
        }
        (return_value, res.is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::AtomicArray;

    #[test]
    fn test_atomic_array_len() {
        let arr: AtomicArray<u8, 8> = AtomicArray::new();
        assert_eq!(0, arr.len());
        assert!(arr.is_empty());

        let mut x: u8 = 100;
        let x_ptr: *mut u8 = &mut x;
        arr.push(x_ptr);

        assert_eq!(1, arr.len());

        let mut y: u8 = 200;
        let y_ptr: *mut u8 = &mut y;
        arr.insert(4, y_ptr);

        assert_eq!(2, arr.len());
    }
}
