use async_std::stream::Cloned;
use dashmap::DashMap;
use futures::future::{join_all, BoxFuture, FutureExt};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;
use tokio::task;

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

#[derive(Debug, Error)]
pub enum WaCustomError {
    #[error("Failed to create the database")]
    CreateDatabaseFailed,

    #[error("Failed to upsert vectors")]
    UpsertFailed,
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
