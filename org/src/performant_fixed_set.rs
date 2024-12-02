use std::collections::HashSet;
use std::time::Instant;

struct PerformantFixedSet {
    static_random: u32,
    buckets: [u128; 64]
}

impl PerformantFixedSet {
    fn new(static_random: u32) -> Self {
        PerformantFixedSet {
            static_random,
            buckets: [0; 64]
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

    fn insert(&mut self, value: u32) {
        let index = self.get_bucket_index(value);
        let bit_position = self.custom_bit_mapping(value);
        
        self.buckets[index] |= 1u128 << bit_position;
    }

    fn is_member(&self, value: u32) -> bool {
        let index = self.get_bucket_index(value);
        let bit_position = self.custom_bit_mapping(value);
        
        (self.buckets[index] & (1u128 << bit_position)) != 0
    }
}

fn benchmark_insert(total_elements: usize) -> std::time::Duration {
    let static_random = 0x12345678;
    let mut fixed_set = PerformantFixedSet::new(static_random);
    
    let start = Instant::now();
    for i in 0..total_elements {
        fixed_set.insert(i as u32);
    }
    start.elapsed()
}

fn benchmark_hashset_insert(total_elements: usize) -> std::time::Duration {
    let mut hash_set = HashSet::new();
    
    let start = Instant::now();
    for i in 0..total_elements {
        hash_set.insert(i);
    }
    start.elapsed()
}

fn benchmark_lookup(total_elements: usize) -> (std::time::Duration, usize) {
    let static_random = 0x12345678;
    let mut fixed_set = PerformantFixedSet::new(static_random);
    
    // Populate the set
    for i in 0..total_elements {
        fixed_set.insert(i as u32);
    }
    
    // Benchmark lookups
    let start = Instant::now();
    let lookups = (0..total_elements)
        .filter(|&i| fixed_set.is_member(i as u32))
        .count();
    let duration = start.elapsed();
    
    (duration, lookups)
}

fn benchmark_hashset_lookup(total_elements: usize) -> (std::time::Duration, usize) {
    let mut hash_set = HashSet::new();
    
    // Populate the set
    for i in 0..total_elements {
        hash_set.insert(i);
    }
    
    // Benchmark lookups
    let start = Instant::now();
    let lookups = (0..total_elements)
        .filter(|&i| hash_set.contains(&i))
        .count();
    let duration = start.elapsed();
    
    (duration, lookups)
}

fn main() {
    let total_elements = 5000;
    
    // Insertion Benchmarks
    let fixed_set_insert_time = benchmark_insert(total_elements);
    let hashset_insert_time = benchmark_hashset_insert(total_elements);
    
    // Lookup Benchmarks
    let (fixed_set_lookup_time, fixed_set_lookups) = benchmark_lookup(total_elements);
    let (hashset_lookup_time, hashset_lookups) = benchmark_hashset_lookup(total_elements);
    
    println!("Benchmark Results for {} elements:", total_elements);
    println!("FixedSet Insertion Time:  {:?}", fixed_set_insert_time);
    println!("HashSet Insertion Time:   {:?}", hashset_insert_time);
    println!("FixedSet Lookup Time:     {:?}", fixed_set_lookup_time);
    println!("HashSet Lookup Time:      {:?}", hashset_lookup_time);
    println!("FixedSet Lookups:         {}", fixed_set_lookups);
    println!("HashSet Lookups:          {}", hashset_lookups);
}
