pub struct PerformantFixedSet {
    buckets: [u128; 59], // its a prime number
}

impl PerformantFixedSet {
    pub fn new() -> Self {
        PerformantFixedSet { buckets: [0; 59] }
    }

    pub fn insert(&mut self, value: u64) {
        let index = (value % 59) as usize;
        // let hash = value ^ 0xA5A5A5A5;
        let bit_position = (value % 128) as u8;

        self.buckets[index] |= 1u128 << bit_position;
    }

    pub fn is_member(&self, value: u64) -> bool {
        let index = (value % 59) as usize;
        // let hash = value ^ 0xA5A5A5A5;
        let bit_position = (value % 128) as u8;

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
