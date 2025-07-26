use dashmap::DashMap;
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use de::FieldValueVisitor;
use schema::MetadataDimensions;
use serde::{Deserialize, Deserializer, Serialize};

pub mod de;
pub mod query_filtering;
pub mod schema;

pub use query_filtering::{Filter, Operator, Predicate, QueryFilterDimensions};
pub use schema::MetadataSchema;

use crate::models::common::generate_level_probs;
use crate::models::types::InternalId;

pub const HIGH_WEIGHT: i32 = 1;

// Global concurrent caches for DP optimizations
static BINARY_CONVERSION_CACHE: OnceLock<DashMap<(u16, usize), Vec<u8>>> = OnceLock::new();
static COMBINATIONS_CACHE: OnceLock<DashMap<Vec<Vec<u16>>, Vec<Vec<u16>>>> = OnceLock::new();
static DIMENSIONS_CACHE: OnceLock<DashMap<(String, Option<String>), Vec<MetadataDimensions>>> =
    OnceLock::new();

/// Get the binary conversion cache
fn get_binary_cache() -> &'static DashMap<(u16, usize), Vec<u8>> {
    BINARY_CONVERSION_CACHE.get_or_init(|| DashMap::new())
}

/// Get the combinations cache
fn get_combinations_cache() -> &'static DashMap<Vec<Vec<u16>>, Vec<Vec<u16>>> {
    COMBINATIONS_CACHE.get_or_init(|| DashMap::new())
}

/// Get the dimensions cache
fn get_dimensions_cache() -> &'static DashMap<(String, Option<String>), Vec<MetadataDimensions>> {
    DIMENSIONS_CACHE.get_or_init(|| DashMap::new())
}

/// Clear all global caches (useful for testing or memory management)
pub fn clear_all_caches() {
    if let Some(cache) = BINARY_CONVERSION_CACHE.get() {
        cache.clear();
    }
    if let Some(cache) = COMBINATIONS_CACHE.get() {
        cache.clear();
    }
    if let Some(cache) = DIMENSIONS_CACHE.get() {
        cache.clear();
    }
}

/// Get cache statistics for monitoring
pub fn get_cache_stats() -> (usize, usize, usize) {
    let binary_size = BINARY_CONVERSION_CACHE.get().map(|c| c.len()).unwrap_or(0);
    let combinations_size = COMBINATIONS_CACHE.get().map(|c| c.len()).unwrap_or(0);
    let dimensions_size = DIMENSIONS_CACHE.get().map(|c| c.len()).unwrap_or(0);
    (binary_size, combinations_size, dimensions_size)
}

#[derive(Debug, Clone)]
pub enum Error {
    InvalidField(String),
    InvalidFieldCardinality(String),
    InvalidFieldValue(String),
    InvalidFieldValues(String),
    InvalidMetadataSchema,
    UnsupportedFilter(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidField(msg) => write!(f, "Invalid field: {msg}"),
            Self::InvalidFieldCardinality(msg) => write!(f, "Invalid field cardinality: {msg}"),
            Self::InvalidFieldValue(msg) => write!(f, "Invalid field value: {msg}"),
            Self::InvalidFieldValues(msg) => write!(f, "Invalid field values: {msg}"),
            Self::InvalidMetadataSchema => write!(f, "Invalid metadata schema"),
            Self::UnsupportedFilter(msg) => write!(f, "Unsupported filter: {msg}"),
        }
    }
}

/// Returns power of 2 that's nearest to the number, rounded up
fn nearest_power_of_two(n: u16) -> Option<u8> {
    let powers: Vec<u16> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    for (i, p) in powers.iter().enumerate() {
        if *p >= n {
            return Some(i as u8);
        }
    }
    None
}

/// Converts a number from decimal to binary, represented as a vector
/// of u8
fn decimal_to_binary_vec(num: u16, size: usize) -> Vec<u8> {
    // Create a vector with specified size, initialized with
    // zeros. Assuming that the binary representation of `num` will
    // fit in `size` i.e. 2^size > num
    let mut result = vec![0; size];
    // Convert decimal to binary, starting from the rightmost position
    let mut n = num;
    for i in (0..size).rev() {
        result[i] = (n & 1) as u8;
        n >>= 1;
    }
    result
}

/// Memoized version of decimal_to_binary_vec for better performance
pub fn decimal_to_binary_vec_memoized(num: u16, size: usize) -> Vec<u8> {
    let cache = get_binary_cache();

    // Check if result is already cached
    if let Some(cached_result) = cache.get(&(num, size)) {
        return cached_result.clone();
    }

    // Compute the result
    let result = decimal_to_binary_vec(num, size);

    // Cache the result
    cache.insert((num, size), result.clone());

    result
}

/// Optimized combination generation using Dynamic Programming with global caching
/// This version avoids redundant work by building combinations incrementally
/// and reusing intermediate results from a global concurrent cache
pub fn gen_combinations_optimized(vs: &Vec<Vec<u16>>) -> Vec<Vec<u16>> {
    if vs.is_empty() {
        return vec![];
    }

    let cache = get_combinations_cache();

    // Check if result is already cached
    if let Some(cached_result) = cache.get(vs) {
        return cached_result.clone();
    }

    // Use DP table to store intermediate combinations
    // dp[i] represents all combinations using vectors from index 0 to i
    let mut dp: Vec<Vec<Vec<u16>>> = Vec::with_capacity(vs.len());

    // Initialize with combinations from the first vector
    let mut initial_combinations = Vec::new();
    for &item in &vs[0] {
        initial_combinations.push(vec![item]);
    }
    dp.push(initial_combinations);

    // Build combinations incrementally using DP
    for i in 1..vs.len() {
        let mut new_combinations = Vec::new();
        let current_vector = &vs[i];

        // For each existing combination, extend it with each element from current vector
        for combination in &dp[i - 1] {
            for &item in current_vector {
                let mut new_combination = combination.clone();
                new_combination.push(item);
                new_combinations.push(new_combination);
            }
        }

        dp.push(new_combinations);
    }

    // Get the final result
    let result = dp.pop().unwrap_or_default();

    // Cache the result
    cache.insert(vs.clone(), result.clone());

    result
}

type FieldName = String;

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[non_exhaustive]
pub enum FieldValue {
    Int(i32),
    String(String),
    // @TODO: Add support for float
}

impl FieldValue {
    fn type_as_str(&self) -> &str {
        match self {
            Self::Int(_) => "int",
            Self::String(_) => "string",
        }
    }
}

impl Serialize for FieldValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Int(i) => serializer.serialize_i32(*i),
            Self::String(s) => serializer.serialize_str(s),
        }
    }
}

impl<'de> Deserialize<'de> for FieldValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(FieldValueVisitor)
    }
}

pub type MetadataFields = HashMap<FieldName, FieldValue>;

/// Returns metadata dimensions based on the fields
///
/// Note that the base dimensions will be implicitly included as the
/// first item in the returned vector
pub fn fields_to_dimensions(
    schema: &MetadataSchema,
    metadata_fields: Option<&MetadataFields>,
) -> Result<Vec<MetadataDimensions>, Error> {
    let mut result = vec![];
    // First add base dimensions
    result.push(schema.base_dimensions());

    if let Some(fields) = metadata_fields {
        let weighted_dims = schema.weighted_dimensions(fields, HIGH_WEIGHT)?;
        for wd in weighted_dims.into_iter() {
            result.push(wd);
        }
    }

    Ok(result)
}

/// Calculates level probs for pseudo replica nodes that are added at
/// the time of index creation for collections that have metadata
/// schema.
///
/// Similar to the `generate_level_probs` fn, the `num_levels` arg
/// represents no. of HNSW levels, hence the result includes an
/// additional level 0.
///
/// The caller must ensure that `num_pseudo_nodes` is not equal to 0,
/// otherwise this fn will panic.
pub fn pseudo_level_probs(num_levels: u8, num_pseudo_nodes: u16) -> Vec<(f64, u8)> {
    // @NOTE: It's ok to case u32 to u8 below as log to the base 10 of
    // u16::MAX is only 4.
    let mut num_higher_levels = (num_pseudo_nodes.ilog10() + 1) as u8;
    // Find lower levels, handling the case where `num_higher_levels`
    // happens to be greater than `num_levels`. In that case, all
    // levels are considered lower.
    let diff = num_levels.overflowing_sub(num_higher_levels);
    let num_lower_levels = match diff {
        (d, false) => d,
        (_, true) => {
            num_higher_levels = 0;
            num_levels
        }
    };
    let mut result = Vec::with_capacity(num_levels as usize);
    if num_higher_levels > 0 {
        let higher_probs = generate_level_probs(10.0, num_higher_levels);
        for (prob, level) in higher_probs {
            if level == 0 {
                continue;
            }
            result.push((prob, num_lower_levels + level));
        }
    }
    for i in (0..=num_lower_levels).rev() {
        result.push((0.0, i));
    }
    result
}

/// Returns vector values for pseudo node
pub fn pseudo_node_vector(num_dims: usize) -> Vec<f32> {
    vec![0.0; num_dims]
}

/// Returns the internal id for pseudo root node
pub fn pseudo_root_id() -> InternalId {
    // The last 258 IDs in the generate u32 internal IDs range are
    // reserved for special cases, `u32::MAX` is for root node,
    // `u32::MAX - 1` is for queries, and the range `[u32::MAX -
    // 257, u32::MAX - 2]`
    InternalId::from(u32::MAX - 257)
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_nearest_power_of_two() {
        assert_eq!(2, nearest_power_of_two(3).unwrap());
        assert_eq!(0, nearest_power_of_two(1).unwrap());
        assert_eq!(6, nearest_power_of_two(33).unwrap());
        assert_eq!(10, nearest_power_of_two(1023).unwrap());
        assert_eq!(10, nearest_power_of_two(1024).unwrap());
        assert!(nearest_power_of_two(1025).is_none());
    }

    #[test]
    fn test_decimal_to_binary_vec() {
        assert_eq!(vec![0, 1, 1, 1], decimal_to_binary_vec(7, 4));
        assert_eq!(vec![0, 0, 1, 1], decimal_to_binary_vec(3, 4));
    }

    #[test]
    fn test_gen_combinations() {
        let vs = vec![vec![1, 2, 3], vec![4, 5]];
        let cs = gen_combinations_optimized(&vs)
            .into_iter()
            .collect::<HashSet<Vec<u16>>>();
        let expected: Vec<Vec<u16>> = vec![
            vec![1, 4],
            vec![1, 5],
            vec![2, 4],
            vec![2, 5],
            vec![3, 4],
            vec![3, 5],
        ];
        let e = expected.into_iter().collect::<HashSet<Vec<u16>>>();
        assert_eq!(e, cs);

        let vs = vec![vec![0], vec![1, 2], vec![4, 5]];
        let cs = gen_combinations_optimized(&vs)
            .into_iter()
            .collect::<HashSet<Vec<u16>>>();
        let expected: Vec<Vec<u16>> =
            vec![vec![0, 1, 4], vec![0, 1, 5], vec![0, 2, 4], vec![0, 2, 5]];
        let e = expected.into_iter().collect::<HashSet<Vec<u16>>>();
        assert_eq!(e, cs);

        let vs = vec![vec![0], vec![0], vec![4, 5]];
        let cs = gen_combinations_optimized(&vs)
            .into_iter()
            .collect::<HashSet<Vec<u16>>>();
        let expected: Vec<Vec<u16>> = vec![vec![0, 0, 4], vec![0, 0, 5]];
        let e = expected.into_iter().collect::<HashSet<Vec<u16>>>();
        assert_eq!(e, cs);
    }

    #[test]
    fn test_pseudo_level_probs() {
        let lp = pseudo_level_probs(9, 128);
        let expected = vec![
            (0.999, 9),
            (0.99, 8),
            (0.9, 7),
            (0.0, 6),
            (0.0, 5),
            (0.0, 4),
            (0.0, 3),
            (0.0, 2),
            (0.0, 1),
            (0.0, 0),
        ];
        assert_eq!(expected, lp);
    }

    #[test]
    fn test_global_concurrent_cache() {
        // Clear caches before test
        clear_all_caches();

        // Test binary conversion cache
        let result1 = decimal_to_binary_vec_memoized(5, 4);
        let result2 = decimal_to_binary_vec_memoized(5, 4);
        assert_eq!(result1, result2);
        assert_eq!(result1, vec![0, 1, 0, 1]);

        // Test combinations cache
        let input = vec![vec![1, 2], vec![3, 4]];
        let result1 = gen_combinations_optimized(&input);
        let result2 = gen_combinations_optimized(&input);
        assert_eq!(result1, result2);
        assert_eq!(
            result1,
            vec![vec![1, 3], vec![1, 4], vec![2, 3], vec![2, 4]]
        );

        // Check cache stats
        let (binary_size, combinations_size, dimensions_size) = get_cache_stats();
        assert!(binary_size > 0);
        assert!(combinations_size > 0);
        assert_eq!(dimensions_size, 0); // No dimensions cached yet

        // Clear caches
        clear_all_caches();
        let (binary_size, combinations_size, dimensions_size) = get_cache_stats();
        assert_eq!(binary_size, 0);
        assert_eq!(combinations_size, 0);
        assert_eq!(dimensions_size, 0);
    }
}
