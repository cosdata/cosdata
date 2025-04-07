use std::cmp::{Ord, PartialOrd};
use std::collections::HashMap;
use std::fmt;

use de::FieldValueVisitor;
use schema::MetadataDimensions;
use serde::{Deserialize, Deserializer, Serialize};

pub mod de;
pub mod query_filtering;
pub mod schema;

pub use query_filtering::{Filter, Operator, Predicate, QueryFilterDimensions};
pub use schema::MetadataSchema;

const HIGH_WEIGHT: i32 = 64000;

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

fn gen_combinations(vs: &Vec<Vec<u16>>) -> Vec<Vec<u16>> {
    if vs.is_empty() {
        return vec![];
    }
    // Start with a single empty combination
    let mut combinations = vec![Vec::new()];
    // For each vector in the input
    for v in vs {
        // Create new combinations by extending each existing combination
        // with each element from the current vector
        let mut new_combinations = Vec::new();
        for combination in combinations {
            for item in v {
                // Create a new combination by cloning the existing
                // one and adding the new item
                let mut new_combination = combination.clone();
                new_combination.push(*item);
                new_combinations.push(new_combination);
            }
        }
        // Replace the old combinations with the new ones
        combinations = new_combinations;
    }
    combinations
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
        let cs = gen_combinations(&vs)
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
        let cs = gen_combinations(&vs)
            .into_iter()
            .collect::<HashSet<Vec<u16>>>();
        let expected: Vec<Vec<u16>> =
            vec![vec![0, 1, 4], vec![0, 1, 5], vec![0, 2, 4], vec![0, 2, 5]];
        let e = expected.into_iter().collect::<HashSet<Vec<u16>>>();
        assert_eq!(e, cs);

        let vs = vec![vec![0], vec![0], vec![4, 5]];
        let cs = gen_combinations(&vs)
            .into_iter()
            .collect::<HashSet<Vec<u16>>>();
        let expected: Vec<Vec<u16>> = vec![vec![0, 0, 4], vec![0, 0, 5]];
        let e = expected.into_iter().collect::<HashSet<Vec<u16>>>();
        assert_eq!(e, cs);
    }
}
