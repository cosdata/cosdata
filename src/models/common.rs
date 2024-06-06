use async_std::stream::Cloned;
use dashmap::DashMap;
use futures::future::{join_all, BoxFuture, FutureExt};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use thiserror::Error;
use tokio::task;

use super::rpc::{Vector, VectorIdValue};
use super::types::VectorId;

pub struct CosResult {
    pub dotprod: i32,
    pub premag_a: i32,
    pub premag_b: i32,
}

// Function to convert a sequence of bits to an integer value
fn bits_to_integer(bits: &[i32], size: usize) -> u32 {
    let mut result: u32 = 0;
    for i in 0..size {
        result = (result << 1) | (bits[i] as u32);
    }
    result
}

fn x_function(value: u32) -> i32 {
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
        _ => -1, // Invalid input
    }
}

fn shift_and_accumulate(value: u32) -> i32 {
    let mut result: i32 = 0;
    result += x_function(15 & (value >> 0));
    result += x_function(15 & (value >> 4));
    result += x_function(15 & (value >> 8));
    result += x_function(15 & (value >> 12));
    result
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(vec: &[f32]) -> f32 {
    vec.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let cs = dot_product(a, b) / (magnitude(a) * magnitude(b));
    // println!("cs: {:?}", cs);
    cs
}

pub fn compute_cosine_similarity(a: &[i32], b: &[i32], size: usize) -> CosResult {
    let value1 = bits_to_integer(a, size);
    let value2 = bits_to_integer(b, size);

    let dotprod = shift_and_accumulate(value1 & value2);
    let premag_a = shift_and_accumulate(value1);
    let premag_b = shift_and_accumulate(value2);

    CosResult {
        dotprod,
        premag_a,
        premag_b,
    }
}

pub fn cosine_coalesce(x: &[Vec<i32>], y: &[Vec<i32>]) -> f64 {
    if x.len() != y.len() {
        panic!("Input vectors must have the same length");
    }

    let mut results = Vec::new();

    for (sub_x, sub_y) in x.iter().zip(y.iter()) {
        let cs = compute_cosine_similarity(sub_x, sub_y, 16);
        results.push(cs);
    }

    let summed = sum_components(&results);

    f64::from(summed.dotprod)
        / (f64::sqrt(f64::from(summed.premag_a)) * f64::sqrt(f64::from(summed.premag_b)))
}

pub fn cosine_sim_unsigned(x: &Vec<u32>, y: &Vec<u32>) -> f64 {
    let mut acc = CosResult {
        dotprod: 0,
        premag_a: 0,
        premag_b: 0,
    };
    for (value1, value2) in x.iter().zip(y.iter()) {
        let dotprod = shift_and_accumulate(value1 & value2);
        let premag_a = shift_and_accumulate(*value1);
        let premag_b = shift_and_accumulate(*value2);

        acc.dotprod += dotprod;
        acc.premag_a += premag_a;
        acc.premag_b += premag_b;
    }

    f64::from(acc.dotprod)
        / (f64::sqrt(f64::from(acc.premag_a)) * f64::sqrt(f64::from(acc.premag_b)))
}

fn sum_components(results: &[CosResult]) -> CosResult {
    let mut acc = CosResult {
        dotprod: 0,
        premag_a: 0,
        premag_b: 0,
    };

    for res in results {
        acc.dotprod += res.dotprod;
        acc.premag_a += res.premag_a;
        acc.premag_b += res.premag_b;
    }

    acc
}

fn to_float_flag(x: f32) -> i32 {
    if x >= 0.0 {
        1
    } else {
        0
    }
}

pub fn floats_to_bits(floats: &[f32]) -> Vec<u32> {
    let mut result = vec![0; (floats.len() + 31) / 32];

    for (i, &f) in floats.iter().enumerate() {
        if f >= 0.0 {
            result[i / 32] |= 1 << (i % 32);
        }
    }

    result
}

pub fn quantize(fins: &[f32]) -> Vec<Vec<i32>> {
    let mut quantized = Vec::with_capacity((fins.len() + 15) / 16);
    let mut chunk = Vec::with_capacity(16);

    for &f in fins {
        chunk.push(to_float_flag(f));
        if chunk.len() == 16 {
            quantized.push(chunk.clone());
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        quantized.push(chunk);
    }

    quantized
}

#[derive(Debug, Error, Clone)]
pub enum WaCustomError {
    #[error("Failed to create the database")]
    CreateDatabaseFailed (String),

    #[error("Failed to create the Column family")]
    CreateCFFailed (String),

    #[error("column family read/write failed")]
    CFReadWriteFailed (String),

    #[error("Failed to upsert vectors")]
    UpsertFailed,

    #[error("ColumnFamily not found")]
    CFNotFound,

    #[error("Invalid params in request")]
    InvalidParams, 
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

pub fn find_value(x: f64) -> i32 {
    let lst = vec![
        (0.0, 0),
        (0.9, 1),
        (0.99, 2),
        (0.999, 3),
        (0.9999, 4),
        (0.99999, 5),
    ];
    let reversed_list = lst.iter().rev();
    match reversed_list.clone().find(|(value, _)| x >= *value) {
        Some((_, index)) => *index,
        None => panic!("No matching element found"),
    }
}

pub fn add_option_vecs(
    a: &Option<Vec<(VectorId, f32)>>,
    b: &Option<Vec<(VectorId, f32)>>,
) -> Option<Vec<(VectorId, f32)>> {
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