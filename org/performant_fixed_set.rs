use std::collections::HashSet;
use std::time::Instant;

struct PerformantFixedSet {
    // Static random number to XOR with input
    static_random: u32,
    buckets: [u128; 32]
}

impl PerformantFixedSet {
    fn new(static_random: u32) -> Self {
        PerformantFixedSet {
            static_random,
            buckets: [0; 32]
        }
    }

    fn get_bucket_index(&self, value: u32) -> usize {
        // XOR the value with static random number, then modulo 32
        ((value ^ self.static_random) % 32) as usize
    }

    fn insert(&mut self, value: u32) {
        let index = self.get_bucket_index(value);
        let bit_position = value % 128; // Use modulo to fit within 128 bits
        
        // Set the corresponding bit
        self.buckets[index] |= 1 << bit_position;
    }

    fn is_member(&self, value: u32) -> bool {
        let index = self.get_bucket_index(value);
        let bit_position = value % 128;
        
        // Check if the bit is set
        (self.buckets[index] & (1 << bit_position)) != 0
    }
}

fn main() {
    // Use a fixed random number for consistent distribution
    let static_random = 0x12345678;

    let total_elements = 300;
    let test_elements = 300;

    // Benchmark HashSet
    let mut hash_set = HashSet::new();
    let start = Instant::now();
    for i in 0..total_elements {
        hash_set.insert(i);
    }
    let hash_set_insert_time = start.elapsed();

    let start = Instant::now();
    let hash_set_lookups = (0..test_elements)
        .filter(|&i| hash_set.contains(&i))
        .count();
    let hash_set_lookup_time = start.elapsed();

    // Benchmark FixedSet
    let mut fixed_set = PerformantFixedSet::new(static_random);
    let start = Instant::now();
    for i in 0..total_elements {
        fixed_set.insert(i);
    }
    let fixed_set_insert_time = start.elapsed();

    let start = Instant::now();
    let fixed_set_lookups = (0..test_elements)
        .filter(|&i| fixed_set.is_member(i))
        .count();
    let fixed_set_lookup_time = start.elapsed();

    println!("HashSet Insertion Time:  {:?}", hash_set_insert_time);
    println!("FixedSet Insertion Time: {:?}", fixed_set_insert_time);
    println!("HashSet Lookup Time:     {:?}", hash_set_lookup_time);
    println!("FixedSet Lookup Time:    {:?}", fixed_set_lookup_time);
    println!("HashSet Lookups:         {}", hash_set_lookups);
    println!("FixedSet Lookups:        {}", fixed_set_lookups);
}
