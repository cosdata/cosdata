use super::buffered_io::BufIoError;
use super::cache_loader::HNSWIndexCache;
use super::prob_node::SharedLatestNode;
use super::types::{InternalId, MetricResult, ReplicaNodeKind};
use crate::distance::DistanceError;
use crate::indexes::hnsw::HNSWIndex;
use crate::metadata;
use crate::quantization::QuantizationError;
use sha2::{Digest, Sha256};
use std::collections::hash_map::{DefaultHasher, Entry};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::{fmt, thread};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub fn dot_product_u8_avx2_fma(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());

    let mut dot_product: u64 = 0;

    // Process 32 elements at a time
    let mut i = 0;
    while i + 32 <= a.len() {
        unsafe {
            // Load 32 elements from each array into AVX2 registers
            let va1 = _mm256_loadu_si256(a[i..].as_ptr() as *const __m256i);
            let vb1 = _mm256_loadu_si256(b[i..].as_ptr() as *const __m256i);

            // Unpack to 16-bit integers
            let va1_lo = _mm256_unpacklo_epi8(va1, _mm256_setzero_si256());
            let vb1_lo = _mm256_unpacklo_epi8(vb1, _mm256_setzero_si256());
            let prod1_lo = _mm256_madd_epi16(va1_lo, vb1_lo);

            let va1_hi = _mm256_unpackhi_epi8(va1, _mm256_setzero_si256());
            let vb1_hi = _mm256_unpackhi_epi8(vb1, _mm256_setzero_si256());
            let prod1_hi = _mm256_madd_epi16(va1_hi, vb1_hi);

            // Horizontal add within 256-bit registers
            let sum1 = _mm256_add_epi32(prod1_lo, prod1_hi);
            let sum2 = _mm256_permute4x64_epi64(sum1, 0b11011000); // permute for horizontal add
            let sum3 = _mm256_hadd_epi32(sum2, sum2);
            let sum4 = _mm256_hadd_epi32(sum3, sum3);

            // Extract result to scalar
            dot_product += _mm256_extract_epi64(sum4, 0) as u64;
        }
        i += 32;
    }

    // Handle remaining elements
    while i < a.len() {
        dot_product += a[i] as u64 * b[i] as u64;
        i += 1;
    }

    dot_product
}

#[allow(dead_code)]
pub struct CosResult {
    pub dotprod: i32,
    pub premag_a: i32,
    pub premag_b: i32,
}
#[allow(dead_code)]
pub fn dot_product_u8_xxx(src: &[(u8, u8)], dst: &mut [u64]) {
    let dst_known_bounds = &mut dst[0..src.len()];
    let size = 8;
    let len = src.len();
    // Process chunks of 8
    let mut i = 0;
    while i + size <= len {
        dst_known_bounds[i] = ((src[0].0) * (src[0].1)) as u64;
        dst_known_bounds[i + 1] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 2] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 3] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 4] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 5] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 6] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 7] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        i += size;
    }
    // Handle remaining elements
    while i < len {
        dst_known_bounds[i] = (src[i].0 as u64) * (src[i].1 as u64);
        i += 1;
    }
}
#[allow(dead_code)]
pub fn dot_product_f32_xxx(src: &[(f32, f32)], dst: &mut [f32]) {
    let dst_known_bounds = &mut dst[0..src.len()];
    let size = 4;
    let len = src.len();
    // Process chunks of 8
    let mut i = 0;
    while i + size <= len {
        dst_known_bounds[i] = (src[0].0) * (src[0].1);
        dst_known_bounds[i + 1] = (src[i + 1].0) * (src[i + 1].1);
        dst_known_bounds[i + 2] = (src[i + 1].0) * (src[i + 1].1);
        dst_known_bounds[i + 3] = (src[i + 1].0) * (src[i + 1].1);
        i += size;
    }
    // Handle remaining elements
    while i < len {
        dst_known_bounds[i] = (src[i].0) * (src[i].1);
        i += 1;
    }
}

#[allow(dead_code)]
pub fn dot_product_f32_chunk(src: &[(f32, f32)], _dst: &mut [f32]) -> f32 {
    let mut d: f32 = 0.0;
    let size = 4;

    // Process chunks of 4
    for chunk in src.chunks_exact(size) {
        let mut local_sum: f32 = 0.0;
        local_sum += chunk[0].0 * chunk[0].1;
        local_sum += chunk[1].0 * chunk[1].1;
        local_sum += chunk[2].0 * chunk[2].1;
        local_sum += chunk[3].0 * chunk[3].1;
        d += local_sum;
    }

    // Handle remaining elements
    for &(a, b) in src.chunks_exact(size).remainder() {
        d += a * b;
    }

    d
}
#[allow(dead_code)]
pub fn dot_product_u8_chunk(src: &[(u8, u8)]) -> u64 {
    let mut d: u64 = 0;
    let size = 8;

    // Process chunks of 8
    for chunk in src.chunks_exact(size) {
        let mut local_sum: u64 = 0;
        local_sum += (chunk[0].0 as u64) * (chunk[0].1 as u64);
        local_sum += (chunk[1].0 as u64) * (chunk[1].1 as u64);
        local_sum += (chunk[2].0 as u64) * (chunk[2].1 as u64);
        local_sum += (chunk[3].0 as u64) * (chunk[3].1 as u64);
        local_sum += (chunk[4].0 as u64) * (chunk[4].1 as u64);
        local_sum += (chunk[5].0 as u64) * (chunk[5].1 as u64);
        local_sum += (chunk[6].0 as u64) * (chunk[6].1 as u64);
        local_sum += (chunk[7].0 as u64) * (chunk[7].1 as u64);
        d += local_sum;
    }

    // Handle remaining elements
    for &(a, b) in src.chunks_exact(size).remainder() {
        d += (a as u64) * (b as u64);
    }

    d
}
#[allow(dead_code)]
pub fn dot_product_a(src: &[(f32, f32)], dst: &mut [f32]) -> f32 {
    let mut d: f32 = 0.0;
    for (_dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
        d += src_sample.0 * src_sample.1;
    }
    d
}

#[allow(dead_code)]
pub fn dot_product_b(src: &[(f32, f32)], dst: &mut [f32]) {
    for (dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
        *dst_sample = src_sample.0 * src_sample.1;
    }
}

#[allow(dead_code)]
pub fn dot_product_u8(src: &[(u8, u8)]) -> u64 {
    src.iter().map(|&(a, b)| (a as u64) * (b as u64)).sum()
}

#[allow(dead_code)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[allow(dead_code)]
fn magnitude(vec: &[f32]) -> f32 {
    vec.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

#[allow(dead_code)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b) / (magnitude(a) * magnitude(b))
}

#[allow(dead_code)]
pub fn get_magnitude_plus_quantized_vec111(quant_vec: Vec<Vec<u32>>, _size: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(quant_vec.len());

    for vecx in quant_vec {
        let premag: u32 = vecx.iter().fold(0, |acc, &val| acc + val.count_ones());
        result.push(premag);
    }

    result
}

#[allow(dead_code)]
pub fn get_magnitude_plus_quantized_vec(quant_vec: &[Vec<u32>], _size: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(quant_vec.len());

    for vecx in quant_vec {
        let premag: usize = vecx
            .iter()
            .fold(0, |acc, &val| acc + val.count_ones() as usize);
        result.push(premag);
    }

    result
}

#[inline]
fn to_float_flag(x: f32, bits_per_value: usize, step: f32) -> Vec<bool> {
    let mut n = ((x + 1.0) / step).floor() as usize;
    let mut result = vec![false; bits_per_value];
    // Fill the vector with bits from the least significant to the most significant
    for i in (0..bits_per_value).rev() {
        result[i] = (n & 1) == 1;
        n >>= 1;
    }

    result
}

pub fn quantize_to_u8_bits(fins: &[f32], resolution: u8) -> Vec<Vec<u8>> {
    let bits_per_value = resolution as usize;
    let parts = 2_usize.pow(bits_per_value as u32);
    let step = 2.0 / parts as f32;
    let u8s_per_value = (fins.len() + 7) / 8;
    let mut quantized: Vec<Vec<u8>> = vec![Vec::with_capacity(u8s_per_value); bits_per_value];

    let mut current_u8s: Vec<u8> = vec![0; bits_per_value];
    let mut bit_index: usize = 0;

    for &f in fins {
        let flags = to_float_flag(f, bits_per_value, step);

        for bit_position in 0..bits_per_value {
            if flags[bit_position] {
                current_u8s[bit_position] |= 1 << bit_index;
            }
        }
        bit_index += 1;

        if bit_index == 8 {
            for bit_position in 0..bits_per_value {
                quantized[bit_position].push(current_u8s[bit_position]);
                current_u8s[bit_position] = 0;
            }
            bit_index = 0;
        }
    }

    // Push remaining bits if not a multiple of 8
    if bit_index > 0 {
        for bit_position in 0..bits_per_value {
            quantized[bit_position].push(current_u8s[bit_position]);
        }
    }

    quantized
}

#[derive(Debug, Clone)]
pub enum WaCustomError {
    DatabaseError(String),
    SerializationError(String),
    UpsertFailed,
    InvalidParams,
    LockError(String),
    QuantizationMismatch,
    LazyLoadingError(String),
    TrainingFailed,
    Untrained,
    CalculationError,
    FsError(String),
    DeserializationError(String),
    // put it in `Arc` to make it cloneable
    BufIo(Arc<BufIoError>),
    MetadataError(metadata::Error),
    NotFound(String),
    ConfigError(String),
    NotImplemented(String),
    InvalidData(String),
}

impl std::error::Error for WaCustomError {}

impl fmt::Display for WaCustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaCustomError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            WaCustomError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            WaCustomError::UpsertFailed => write!(f, "Failed to upsert vectors"),
            WaCustomError::InvalidParams => write!(f, "Invalid params in request"),
            WaCustomError::LockError(msg) => write!(f, "Lock error: {}", msg),
            WaCustomError::QuantizationMismatch => write!(f, "Quantization mismatch"),
            WaCustomError::LazyLoadingError(msg) => write!(f, "Lazy loading error: {}", msg),
            WaCustomError::TrainingFailed => write!(f, "Training failed"),
            WaCustomError::Untrained => write!(f, "Untrained"),
            WaCustomError::CalculationError => write!(f, "Calculation error"),
            WaCustomError::FsError(err) => write!(f, "FS error: {}", err),
            WaCustomError::DeserializationError(err) => write!(f, "Deserialization error: {}", err),
            WaCustomError::BufIo(err) => write!(f, "Buffer IO error: {}", err),
            WaCustomError::MetadataError(err) => write!(f, "Metadata error: {}", err),
            WaCustomError::NotFound(msg) => write!(f, "{} Not Found!", msg),
            WaCustomError::ConfigError(msg) => write!(f, "{} Config file reading error: ", msg),
            WaCustomError::NotImplemented(msg) => write!(f, "Not Implemented: {}", msg),
            WaCustomError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl From<QuantizationError> for WaCustomError {
    fn from(value: QuantizationError) -> Self {
        match value {
            QuantizationError::InvalidInput(_) => WaCustomError::InvalidParams,
            QuantizationError::TrainingFailed => WaCustomError::TrainingFailed,
            QuantizationError::Untrained => WaCustomError::Untrained,
        }
    }
}

impl From<DistanceError> for WaCustomError {
    fn from(value: DistanceError) -> Self {
        match value {
            DistanceError::StorageMismatch => WaCustomError::QuantizationMismatch,
            DistanceError::CalculationError => WaCustomError::CalculationError,
        }
    }
}

impl From<BufIoError> for WaCustomError {
    fn from(error: BufIoError) -> Self {
        Self::BufIo(Arc::new(error))
    }
}

impl From<lmdb::Error> for WaCustomError {
    fn from(error: lmdb::Error) -> Self {
        Self::DatabaseError(error.to_string())
    }
}

#[allow(dead_code)]
pub fn hash_float_vec(vec: Vec<f32>) -> Vec<u8> {
    // Create a new hasher instance
    let mut hasher = Sha256::new();

    // Convert the Vec<f32> to a byte representation
    for &num in &vec {
        // Convert each f32 to its byte representation and update the hasher
        hasher.update(num.to_le_bytes());
    }

    // Finalize the hash and return the result as a Vec<u8>
    hasher.finalize().to_vec()
}

pub fn get_max_insert_level(x: f64, levels: &[(f64, u8)]) -> u8 {
    let lst = levels.iter();
    match lst.clone().find(|(value, _)| x >= *value) {
        Some((_, index)) => *index,
        None => panic!("No matching element found"),
    }
}

pub fn remove_duplicates_and_filter(
    _hnsw_index: &HNSWIndex,
    vec: Vec<(SharedLatestNode, MetricResult)>,
    k: Option<usize>,
    cache: &HNSWIndexCache,
) -> Vec<(InternalId, MetricResult)> {
    let mut seen = HashSet::new();
    let mut collected = vec
        .into_iter()
        .filter_map(|(lazy_item_latest_ptr, similarity)| {
            let lazy_item = unsafe { &*lazy_item_latest_ptr }.latest;
            let node = unsafe { &*lazy_item }.try_get_data(cache).unwrap();
            let replica_id = node.get_id();
            if !seen.insert(replica_id) {
                return None;
            }
            if *replica_id == u32::MAX {
                return None;
            }
            if let ReplicaNodeKind::Pseudo = node.replica_node_kind() {
                return None;
            }
            Some((replica_id, similarity))
        })
        .collect::<Vec<_>>();

    collected.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
    if let Some(k) = k {
        collected.truncate(5 * k);
    }
    collected
}

/// Returns inverse of probabilities for every HNSW level
///
/// Note that the arg `num_levels` represents HNSW levels, hence level
/// 0 gets implicitly added to the result i.e. if num_levels = 9, then
/// the result will be a vector of size 10 with the last element
/// corresponding to level 0, for which the inverse probability will
/// be 0
pub fn generate_level_probs(x: f64, num_levels: u8) -> Vec<(f64, u8)> {
    let mut result = Vec::new();
    for n in (0..=num_levels).rev() {
        let first_item = 1.0 - x.powi(-(n as i32));
        let second_item = n;
        result.push((first_item, second_item));
    }
    result
}

//typically skips is 1 while near
#[allow(dead_code)]
pub fn tapered_skips(skips: i8, cur_distance: i8, max_distance: i8) -> i8 {
    // Calculate the distance ratio (0.0 to 1.0)
    let distance_ratio = cur_distance as f32 / max_distance as f32;

    // Use match expression for efficient logic based on distance ratio
    match distance_ratio {
        ratio if ratio < 0.25 => skips,
        ratio if ratio < 0.5 => skips * 2,
        ratio if ratio < 0.75 => skips * 3,
        _ => skips * 4, // Distance ratio >= 0.75
    }
}

#[allow(dead_code)]
pub fn tuple_to_string(tuple: (u32, u32)) -> String {
    format!("{}_{}", tuple.0, tuple.1)
}

type HashTable<K, V> = HashMap<K, V>;

/// This is a custom Hashtable made to use for data variable in Node of InvertedIndex
// #[derive(Clone)]
pub struct TSHashTable<K, V> {
    pub hash_table_list: Vec<Arc<Mutex<HashTable<K, V>>>>,
    pub size: u8,
}

#[cfg(test)]
impl<K: Eq + Hash + Clone + PartialEq, V: Clone + PartialEq> PartialEq for TSHashTable<K, V> {
    fn eq(&self, other: &Self) -> bool {
        let self_list = self.to_list();
        let other_list = other.to_list();

        for (k, v) in &self_list {
            let Some((_, v2)) = other_list.iter().find(|(k2, _)| k2 == k) else {
                return false;
            };
            if v != v2 {
                return false;
            }
        }

        for (k, v) in &other_list {
            let Some((_, v2)) = self_list.iter().find(|(k2, _)| k2 == k) else {
                return false;
            };
            if v != v2 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
impl<K: Eq + Hash + Clone + std::fmt::Debug + Ord, V: Clone + std::fmt::Debug> std::fmt::Debug
    for TSHashTable<K, V>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = self.to_list();
        list.sort_unstable_by_key(|(k, _)| k.clone());
        f.debug_map().entries(list).finish()
    }
}

unsafe impl<K, V> Send for TSHashTable<K, V> {}
unsafe impl<K, V> Sync for TSHashTable<K, V> {}

#[allow(unused)]
impl<K: Eq + Hash, V> TSHashTable<K, V> {
    pub fn new(size: u8) -> Self {
        let hash_table_list = (0..size)
            .map(|_| Arc::new(Mutex::new(HashMap::new())))
            .collect();
        TSHashTable {
            hash_table_list,
            size,
        }
    }

    pub fn hash_key(&self, k: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        (hasher.finish() as usize) % (self.size as usize)
    }

    pub fn insert(&self, k: K, v: V) {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        ht.insert(k, v);
    }

    pub fn delete(&self, k: &K) {
        let index = self.hash_key(k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        ht.remove(k);
    }

    pub fn lookup(&self, k: &K) -> Option<V>
    where
        V: Clone,
    {
        let index = self.hash_key(k);
        let ht = self.hash_table_list[index].lock().unwrap();
        ht.get(k).cloned()
    }

    pub fn mutate<F>(&self, k: K, f: F)
    where
        F: FnOnce(Option<V>) -> Option<V>,
        V: Clone,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        let v = ht.remove(&k);
        let new_v = f(v);
        if let Some(new_v) = new_v {
            ht.insert(k, new_v);
        }
    }

    pub fn get_or_create<F>(&self, k: K, f: F) -> V
    where
        F: FnOnce() -> V,
        V: Clone,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        match ht.get(&k) {
            Some(v) => v.clone(),
            None => {
                let new_v = f();
                ht.insert(k, new_v.clone());
                new_v
            }
        }
    }

    pub fn modify_or_insert<M, I>(&self, k: K, modify: M, insert: I)
    where
        M: FnOnce(&mut V),
        I: FnOnce() -> V,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        ht.entry(k).and_modify(modify).or_insert_with(insert);
    }

    // works exactly like `modify_or_insert`, but takes a value `T`, which is passed to both
    // closures (only one actually runs, so only passed to one of them), which is not possible
    // with `modify_or_insert` because of Rust's borrow checking rules
    pub fn modify_or_insert_with_value<M, I, T>(&self, k: K, v: T, modify: M, insert: I)
    where
        M: FnOnce(T, &mut V),
        I: FnOnce(T) -> V,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        match ht.entry(k) {
            Entry::Occupied(mut entry) => {
                modify(v, entry.get_mut());
            }
            Entry::Vacant(entry) => {
                entry.insert(insert(v));
            }
        }
    }

    // This bool represents whether the key was found in the map or not.
    pub fn get_or_create_with_flag<F>(&self, k: K, f: F) -> (V, bool)
    where
        F: FnOnce() -> V,
        V: Clone,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        match ht.get(&k) {
            Some(v) => (v.clone(), true),
            None => {
                let new_v = f();
                ht.insert(k, new_v.clone());
                (new_v, false)
            }
        }
    }

    // This bool represents whether the key was found in the map or not.
    pub fn get_or_try_create_with_flag<F, E>(&self, k: K, f: F) -> Result<(V, bool), E>
    where
        F: FnOnce() -> Result<V, E>,
        V: Clone,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        match ht.get(&k) {
            Some(v) => Ok((v.clone(), true)),
            None => {
                let new_v = f()?;
                ht.insert(k, new_v.clone());
                Ok((new_v, false))
            }
        }
    }

    pub fn lock_key_and_try<F, R>(&self, k: K, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let index = self.hash_key(&k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        f()
    }

    pub fn with_value<F, R>(&self, k: &K, f: F) -> Option<R>
    where
        F: FnOnce(&V) -> R,
    {
        let index = self.hash_key(k);
        let ht = self.hash_table_list[index].lock().unwrap();
        ht.get(k).map(f)
    }

    pub fn with_value_mut<F, R>(&self, k: &K, f: F) -> Option<R>
    where
        F: Fn(&mut V) -> R,
    {
        let index = self.hash_key(k);
        let mut ht = self.hash_table_list[index].lock().unwrap();
        ht.get_mut(k).map(f)
    }

    pub fn map_m<F>(&self, f: F)
    where
        F: Fn(&K, &V) + Send + Sync + 'static,
        K: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        let f = Arc::new(f);
        let handles: Vec<_> = self
            .hash_table_list
            .iter()
            .map(|ht| {
                let ht = Arc::clone(ht);
                let f = f.clone();
                thread::spawn(move || {
                    let ht = ht.lock().unwrap();
                    for (k, v) in ht.iter() {
                        f(k, v);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    pub fn to_list(&self) -> Vec<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        self.hash_table_list
            .iter()
            .flat_map(|ht| {
                let ht = ht.lock().unwrap();
                ht.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn from_list(size: u8, kv: Vec<(K, V)>) -> Self {
        let tsh = Self::new(size);
        for (k, v) in kv {
            tsh.insert(k, v);
        }
        tsh
    }

    pub fn purge_all(&self) -> Vec<(K, V)> {
        let mut list = Vec::new();
        for ht in &self.hash_table_list {
            let mut ht = ht.lock().unwrap();
            let map = std::mem::take(&mut *ht);
            list.extend(map.into_iter());
        }
        list
    }

    pub fn for_each<F: FnMut(&K, &V)>(&self, mut f: F) {
        for ht in &self.hash_table_list {
            let ht = ht.lock().unwrap();
            ht.iter().for_each(|(k, v)| f(k, v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::generate_level_probs;

    #[test]
    // @NOTE: This test is added mainly for understanding the
    // implementation
    fn test_generate_level_probs() {
        let lp = generate_level_probs(10.0, 9);
        let expected = vec![
            (0.999999999, 9),
            (0.99999999, 8),
            (0.9999999, 7),
            (0.999999, 6),
            (0.99999, 5),
            (0.9999, 4),
            (0.999, 3),
            (0.99, 2),
            (0.9, 1),
            (0.0, 0),
        ];
        assert_eq!(expected, lp);
    }
}
