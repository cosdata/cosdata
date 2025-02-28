use std::sync::{
    atomic::{AtomicU64, Ordering},
    RwLock,
};

use super::{types::FileOffset, versioning::Hash};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PerformantFixedSet {
    pub buckets: Vec<u64>,
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

#[allow(unused)]
pub struct AtomicFixedSet {
    buckets: Vec<AtomicU64>,
}

#[allow(unused)]
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

pub const INVERTED_FIXED_SET_INDEX_SIZE: usize = 8;

// id -> quantized value
#[derive(Debug)]
pub struct VersionedInvertedFixedSetIndex {
    pub current_hash: Hash,
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub exclusives: Vec<RwLock<PerformantFixedSet>>,
    pub bits: Vec<RwLock<PerformantFixedSet>>,
    pub next: RwLock<Option<Box<VersionedInvertedFixedSetIndex>>>,
}

#[cfg(test)]
impl PartialEq for VersionedInvertedFixedSetIndex {
    fn eq(&self, other: &Self) -> bool {
        self.current_hash == other.current_hash
            && *self.serialized_at.read().unwrap() == *other.serialized_at.read().unwrap()
            && self.exclusives.len() == other.exclusives.len()
            && self.exclusives.iter().zip(&other.exclusives).all(
                |(self_exclusive, other_exclusive)| {
                    *self_exclusive.read().unwrap() == *other_exclusive.read().unwrap()
                },
            )
            && self.bits.len() == other.bits.len()
            && self
                .bits
                .iter()
                .zip(&other.bits)
                .all(|(self_bit, other_bit)| {
                    *self_bit.read().unwrap() == *other_bit.read().unwrap()
                })
            && *self.next.read().unwrap() == *other.next.read().unwrap()
    }
}

impl VersionedInvertedFixedSetIndex {
    pub fn new(quantization_bits: u8, version: Hash) -> Self {
        let quantization = 1u32 << quantization_bits;

        let mut exclusive = Vec::with_capacity(quantization as usize);

        for _ in 0..quantization {
            exclusive.push(RwLock::new(PerformantFixedSet::new(
                INVERTED_FIXED_SET_INDEX_SIZE,
            )));
        }

        let mut bit = Vec::with_capacity(quantization_bits as usize);

        for _ in 0..quantization_bits {
            bit.push(RwLock::new(PerformantFixedSet::new(
                (quantization >> 1) as usize * INVERTED_FIXED_SET_INDEX_SIZE,
            )));
        }

        Self {
            exclusives: exclusive,
            bits: bit,
            current_hash: version,
            serialized_at: RwLock::new(None),
            next: RwLock::new(None),
        }
    }

    pub fn insert(&self, version: Hash, quantized_value: u8, vector_id: u32) {
        if self.current_hash != version {
            let next_read_guard = self.next.read().unwrap();
            if let Some(next) = &*next_read_guard {
                return next.insert(version, quantized_value, vector_id);
            }
            drop(next_read_guard);
            let mut next_write_guard = self.next.write().unwrap();
            if let Some(next) = &*next_write_guard {
                return next.insert(version, quantized_value, vector_id);
            }
            let new_next = Self {
                current_hash: version,
                serialized_at: RwLock::new(None),
                exclusives: self
                    .exclusives
                    .iter()
                    .map(|exclusive| RwLock::new(exclusive.read().unwrap().clone()))
                    .collect(),
                bits: self
                    .bits
                    .iter()
                    .map(|bit| RwLock::new(bit.read().unwrap().clone()))
                    .collect(),
                next: RwLock::new(None),
            };
            new_next.insert(version, quantized_value, vector_id);
            *next_write_guard = Some(Box::new(new_next));
            return;
        }

        self.exclusives[quantized_value as usize]
            .write()
            .unwrap()
            .insert(vector_id);
        for (i, bit) in self.bits.iter().enumerate() {
            if (quantized_value & (1u8 << i)) != 0 {
                bit.write().unwrap().insert(vector_id);
            }
        }
    }

    fn search_bits(&self, vector_id: u32) -> Option<u8> {
        {
            let next_guard = self.next.read().unwrap();
            if let Some(next) = &*next_guard {
                return next.search_bits(vector_id);
            }
        }
        let mut index = 0u8;
        for (i, bit) in self.bits.iter().enumerate() {
            if bit.read().unwrap().is_member(vector_id) {
                index |= 1 << i;
            }
        }

        if index == 0 {
            None
        } else {
            Some(index)
        }
    }

    fn get_permutations(num: u8) -> Vec<u8> {
        let mut result = vec![num];
        let mut one_positions = Vec::new();
        let mut n = num;
        let mut pos = 0;

        // Find positions of 1s
        while n > 0 {
            if n & 1 == 1 {
                one_positions.push(pos);
            }
            n >>= 1;
            pos += 1;
        }

        // For each 1 bit, create new numbers by flipping it to 0
        for &pos in &one_positions {
            let mask = !(1 << pos);
            let len = result.len();
            for i in 0..len {
                let new_num = result[i] & mask;
                if new_num > 0 {
                    // Only add if not zero
                    result.push(new_num);
                }
            }
        }

        result.dedup();
        result.sort_unstable();
        result
    }

    pub fn search(&self, vector_id: u32) -> Option<u8> {
        {
            let next_guard = self.next.read().unwrap();
            if let Some(next) = &*next_guard {
                return next.search(vector_id);
            }
        }

        let index = self.search_bits(vector_id)?;
        let found = self.exclusives[index as usize]
            .read()
            .unwrap()
            .is_member(vector_id);
        if found {
            return Some(index);
        }
        let alternative_keys = Self::get_permutations(index);
        for i in alternative_keys {
            let found = self.exclusives[i as usize]
                .read()
                .unwrap()
                .is_member(vector_id);
            if found {
                return Some(i);
            }
        }
        None
    }
}
