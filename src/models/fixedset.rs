pub struct PerformantFixedSet {
    static_random: u32,
    buckets: [u128; 64],
}

impl PerformantFixedSet {
    pub fn new(static_random: u32) -> Self {
        PerformantFixedSet {
            static_random,
            buckets: [0; 64],
        }
    }

    fn get_bucket_index(&self, value: u32) -> usize {
        ((value ^ self.static_random) % 64) as usize
    }

    fn custom_bit_mapping(&self, value: u32) -> u8 {
        let mut hash = value;
        hash = hash.wrapping_mul(0x5bd1e995);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x5bd1e995);

        (hash % 128) as u8
    }

    pub fn insert(&mut self, value: u32) {
        let index = self.get_bucket_index(value);
        let bit_position = self.custom_bit_mapping(value);

        self.buckets[index] |= 1u128 << bit_position;
    }

    pub fn is_member(&self, value: u32) -> bool {
        let index = self.get_bucket_index(value);
        let bit_position = self.custom_bit_mapping(value);

        (self.buckets[index] & (1u128 << bit_position)) != 0
    }
}

// pub struct PerformantFixedSet {
//     buckets: [Vec<u32>; 128],
// }

// impl PerformantFixedSet {
//     #[rustfmt::skip]
//     pub fn new(_: u32) -> Self {
//         PerformantFixedSet {
//             buckets: [
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//                 Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
//             ],
//         }
//     }

//     pub fn insert(&mut self, value: u32) {
//         let index = (value & 127) as usize;
//         match self.buckets[index].binary_search(&value) {
//             Err(pos) => {
//                 self.buckets[index].insert(pos, value);
//             }
//             Ok(_) => {}
//         }
//     }

//     pub fn is_member(&self, value: u32) -> bool {
//         let index = (value & 127) as usize;
//         self.buckets[index].binary_search(&value).is_ok()
//     }

//     pub fn len(&self) -> usize {
//         self.buckets.iter().map(|bucket| bucket.len()).sum()
//     }
// }
