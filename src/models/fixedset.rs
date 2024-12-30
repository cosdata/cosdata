pub struct PerformantFixedSet {
    buckets: [u64; 64], // Changed from 128 to 64 buckets
}

impl PerformantFixedSet {
    #[inline]
    pub fn new() -> Self {
        Self { buckets: [0; 64] }
    }

    #[inline]
    pub fn insert(&mut self, value: u32) {
        let bucket = (value >> 6) & 63; // Changed mask from 127 to 63
        let bit_pos = value & 63; // Same bit position mask
        self.buckets[bucket as usize] |= 1u64 << bit_pos;
    }

    #[inline]
    pub fn is_member(&self, value: u32) -> bool {
        let bucket = (value >> 6) & 63; // Changed mask from 127 to 63
        let bit_pos = value & 63;
        (self.buckets[bucket as usize] & (1u64 << bit_pos)) != 0
    }
}
