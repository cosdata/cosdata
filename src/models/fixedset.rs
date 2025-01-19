use std::sync::atomic::{AtomicU64, Ordering};

pub struct PerformantFixedSet {
    buckets: Vec<u64>,
}

impl PerformantFixedSet {
    #[inline]
    pub fn new(len: usize) -> Self {
        Self {
            buckets: vec![0; len],
        }
    }

    #[inline]
    pub fn insert(&mut self, value: u32) {
        let mask = self.buckets.len() as u32 - 1;
        let bucket = (value >> 6) & mask;
        let bit_pos = value & 0x3f;
        self.buckets[bucket as usize] |= 1u64 << bit_pos;
    }

    #[inline]
    pub fn is_member(&self, value: u32) -> bool {
        let mask = self.buckets.len() as u32 - 1;
        let bucket = (value >> 6) & mask;
        let bit_pos = value & 0x3f;
        (self.buckets[bucket as usize] & (1u64 << bit_pos)) != 0
    }
}

pub struct AtomicFixedSet {
    buckets: Vec<AtomicU64>,
}

impl AtomicFixedSet {
    #[inline]
    pub fn new(len: usize) -> Self {
        Self {
            buckets: (0..len).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    #[inline]
    pub fn insert(&self, value: u32) {
        let mask = self.buckets.len() as u32 - 1;
        let bucket = (value >> 6) & mask;
        let bit_pos = value & 0x3f;
        self.buckets[bucket as usize].fetch_or(1u64 << bit_pos, Ordering::Relaxed);
    }

    #[inline]
    pub fn is_member(&self, value: u32) -> bool {
        let mask = self.buckets.len() as u32 - 1;
        let bucket = (value >> 6) & mask;
        let bit_pos = value & 0x3f;
        (self.buckets[bucket as usize].load(Ordering::Relaxed) & (1u64 << bit_pos)) != 0
    }
}
