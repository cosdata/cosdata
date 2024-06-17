use super::rpc::VectorIdValue;
use super::types::{NodeRef, VectorId};
use crate::models::rpc::Vector;
use crate::models::types::VectorQt;
use async_std::stream::Cloned;
use dashmap::DashMap;
use futures::future::{join_all, BoxFuture, FutureExt};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;
use tokio::task;

pub struct CosResult {
    pub dotprod: i32,
    pub premag_a: i32,
    pub premag_b: i32,
}

// Function to convert a sequence of bits to an integer value
pub fn bits_to_integer(bits: &[u8], size: usize) -> u32 {
    bits.iter().fold(0, |acc, &bit| (acc << 1) | (bit as u32))
}

fn x_function(value: u32) -> u32 {
    match value {
        0 => 0,
        1 => 1,
        2 => 1,
        3 => 2,
        4 => 1,
        5 => 2,
        6 => 2,
        7 => 3,
        8 => 1,
        9 => 2,
        10 => 2,
        11 => 3,
        12 => 2,
        13 => 3,
        14 => 3,
        15 => 4,
        _ => 0, // Invalid input
    }
}

fn shift_and_accumulate(value: u32) -> u32 {
    let mut result: u32 = 0;
    result += x_function(15 & (value >> 0));
    result += x_function(15 & (value >> 4));
    result += x_function(15 & (value >> 8));
    result += x_function(15 & (value >> 12));
    result += x_function(15 & (value >> 16));
    result += x_function(15 & (value >> 20));
    result += x_function(15 & (value >> 24));
    result += x_function(15 & (value >> 28));

    result
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(vec: &[f32]) -> f32 {
    vec.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b) / (magnitude(a) * magnitude(b))
}

pub fn get_magnitude_plus_quantized_vec(bits: Vec<Vec<u8>>, size: usize) -> (f64, Vec<u32>) {
    // Create the quant_vec with the capacity equal to the length of bits
    let mut quant_vec: Vec<u32> = Vec::with_capacity(size);

    // Fill the quant_vec using a loop or iterator
    for bit_vec in &bits {
        quant_vec.push(bits_to_integer(bit_vec, 32));
    }

    let premag: u32 = quant_vec
        .iter()
        .fold(0, |acc, val| acc + shift_and_accumulate(*val));
    let mag = f64::sqrt(f64::from(premag));

    return (mag, quant_vec);
}

pub fn cosine_coalesce(x: &VectorQt, y: &VectorQt, length: usize) -> f32 {
    let mut dot_prod = 0;
    for i in 0..length {
        dot_prod += shift_and_accumulate(x.quant_vec[i] & y.quant_vec[i]);
    }
    let res = f64::from(dot_prod) / (x.mag * y.mag);
    //print!("cosine coalesce {}", res);
    return res as f32;
}

fn to_float_flag(x: f32) -> u8 {
    if x >= 0.0 {
        1
    } else {
        0
    }
}

pub fn quantize_to_u8_bits(fins: &[f32]) -> Vec<Vec<u8>> {
    let mut quantized: Vec<Vec<u8>> = Vec::with_capacity((fins.len() + 31) / 32);
    let mut chunk: Vec<u8> = Vec::with_capacity(32);

    for &f in fins {
        chunk.push(to_float_flag(f));
        if chunk.len() == 32 {
            quantized.push(chunk.clone());
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        quantized.push(chunk);
    }
    //println!("{:?}", quantized);
    quantized
}

#[derive(Debug, Error, Clone)]
pub enum WaCustomError {
    #[error("Failed to create the database")]
    CreateDatabaseFailed(String),

    #[error("Failed to create the Column family")]
    CreateCFFailed(String),

    #[error("column family read/write failed")]
    CFReadWriteFailed(String),

    #[error("Failed to upsert vectors")]
    UpsertFailed,

    #[error("ColumnFamily not found")]
    CFNotFound,

    #[error("Invalid params in request")]
    InvalidParams,

    #[error("Could not load Node")]
    NodeNotFound(String),

    #[error("Pending neighbor encountered")]
    PendingNeighborEncountered(String),
}

pub fn hash_float_vec(vec: Vec<f32>) -> Vec<u8> {
    // Create a new hasher instance
    let mut hasher = Sha256::new();

    // Convert the Vec<f32> to a byte representation
    for &num in &vec {
        // Convert each f32 to its byte representation and update the hasher
        hasher.update(&num.to_le_bytes());
    }

    // Finalize the hash and return the result as a Vec<u8>
    hasher.finalize().to_vec()
}

pub fn get_max_insert_level(x: f64, levels: Arc<Vec<(f64, i32)>>) -> i32 {
    let lst = levels.iter();
    match lst.clone().find(|(value, _)| x >= *value) {
        Some((_, index)) => *index,
        None => panic!("No matching element found"),
    }
}

pub fn add_option_vecs(
    a: &Option<Vec<(NodeRef, f32)>>,
    b: &Option<Vec<(NodeRef, f32)>>,
) -> Option<Vec<(NodeRef, f32)>> {
    match (a, b) {
        (None, None) => None,
        (Some(vec), None) | (None, Some(vec)) => Some(vec.clone()),
        (Some(vec1), Some(vec2)) => {
            let mut combined = vec1.clone();
            combined.extend(vec2.iter().cloned());
            Some(combined)
        }
    }
}

// Function to convert VectorIdValue to VectorId
pub fn convert_value(id_value: VectorIdValue) -> VectorId {
    match id_value {
        VectorIdValue::StringValue(s) => VectorId::Str(s),
        VectorIdValue::IntValue(i) => VectorId::Int(i),
    }
}

// Function to convert VectorId to VectorIdValue
fn convert_id(id: VectorId) -> VectorIdValue {
    match id {
        VectorId::Str(s) => VectorIdValue::StringValue(s),
        VectorId::Int(i) => VectorIdValue::IntValue(i),
    }
}

// Function to convert the Option<Vec<(VectorId, _)>> to Option<Vec<(VectorIdValue, _)>>
pub fn convert_option_vec(
    input: Option<Vec<(VectorId, f32)>>,
) -> Option<Vec<(VectorIdValue, f32)>> {
    input.map(|vec| {
        vec.into_iter()
            .map(|(id, value)| (convert_id(id), value))
            .collect()
    })
}

// Function to convert Vec<Vector> to Vec<(VectorIdValue, Vec<f32>)>
pub fn convert_vectors(vectors: Vec<Vector>) -> Vec<(VectorIdValue, Vec<f32>)> {
    vectors
        .into_iter()
        .map(|vector| (vector.id.clone(), vector.values))
        .collect()
}

pub fn remove_duplicates_and_filter(
    input: Option<Vec<(NodeRef, f32)>>,
) -> Option<Vec<(VectorId, f32)>> {
    if let Some(vec) = input {
        let mut seen = HashSet::new();
        let mut unique_vec = Vec::new();

        for item in vec {
            if let VectorId::Int(ref s) = item.0.prop.id {
                if *s == -1 {
                    continue;
                }
            }

            if seen.insert(item.0.prop.id.clone()) {
                unique_vec.push((item.0.prop.id.clone(), item.1));
            }
        }

        Some(unique_vec)
    } else {
        None
    }
}

pub fn generate_tuples(x: f64) -> Vec<(f64, i32)> {
    let mut result = Vec::new();
    for n in 0..20 {
        let first_item = 1.0 - x.powi(-(n as i32));
        let second_item = n as i32;
        result.push((first_item, second_item));
    }
    result
}

pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    t.hash(&mut hasher);
    hasher.finish()
}

// Extract VectorId values for hashing purposes
pub fn extract_ids(neighbors: &[(VectorId, f32)]) -> Vec<VectorId> {
    neighbors.iter().map(|(id, _)| id.clone()).collect()
}

// Optional: Implement From trait for more idiomatic conversion

impl From<VectorId> for VectorIdValue {
    fn from(vector_id: VectorId) -> Self {
        match vector_id {
            VectorId::Str(s) => VectorIdValue::StringValue(s),
            VectorId::Int(i) => VectorIdValue::IntValue(i),
        }
    }
}

impl From<VectorIdValue> for VectorId {
    fn from(vector_id_value: VectorIdValue) -> Self {
        match vector_id_value {
            VectorIdValue::StringValue(s) => VectorId::Str(s),
            VectorIdValue::IntValue(i) => VectorId::Int(i),
        }
    }
}

pub fn cat_maybes<T>(iter: impl Iterator<Item = Option<T>>) -> Vec<T> {
    iter.flat_map(|maybe| maybe).collect()
}



pub fn tapered_total_hops(hops: u8, cur_level: u8, max_level: u8) -> u8 {
    //div by 2
    if cur_level > max_level >> 1 {
        return hops;
    } else {
        // div by 4
        if cur_level > max_level >> 2 {
            return 3 * (hops >> 2); // 3/4
        } else {
            return hops >> 1; // 1/2
        }
    }
}
//typically skips is 1 while near
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
