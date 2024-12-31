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
        let bit_pos = value & mask;
        self.buckets[bucket as usize] |= 1u64 << bit_pos;
    }

    #[inline]
    pub fn is_member(&self, value: u32) -> bool {
        let mask = self.buckets.len() as u32 - 1;
        let bucket = (value >> 6) & mask;
        let bit_pos = value & mask;
        (self.buckets[bucket as usize] & (1u64 << bit_pos)) != 0
    }
}
