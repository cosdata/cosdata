mod vector_store;

use std::f64;

struct CosResult {
    dotprod: i32,
    premag_a: i32,
    premag_b: i32,
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

fn dot_product(va: &[i32], vb: &[i32]) -> u32 {
    let value1 = bits_to_integer(va, 8);
    let value2 = bits_to_integer(vb, 8);
    shift_and_accumulate(value1 & value2) as u32
}

fn pre_magnitude(v: &[i32]) -> u32 {
    let value = bits_to_integer(v, 8);
    shift_and_accumulate(value) as u32
}

fn magnitude(v: &[i32]) -> f64 {
    let value = bits_to_integer(v, 8);
    let x = shift_and_accumulate(value);
    (x as f64).sqrt()
}

fn cosine_similarity(v1: &[i32], v2: &[i32]) -> f64 {
    let dp = dot_product(v1, v2) as f64;
    let mag1 = magnitude(v1);
    let mag2 = magnitude(v2);
    dp / (mag1 * mag2)
}

fn compute(a: &[i32], b: &[i32], size: usize) -> CosResult {
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

fn main() {
    // Example bit sequence
    let bits1 = [1, 0, 1, 1, 0, 1, 1, 1]; // Represents the binary value 101101
    let bits2 = [1, 0, 0, 1, 0, 0, 0, 1]; // Represents the binary value 101101

    // Compute the cosine similarity
    let result = cosine_similarity(&bits1, &bits2);
    println!("Cosine similarity result: {}", result);

    // Compute the CosResult struct
    let cos_result = compute(&bits1, &bits2, 8);
    println!(
        "CosResult -> dotprod: {}, premagA: {}, premagB: {}",
        cos_result.dotprod, cos_result.premag_a, cos_result.premag_b
    );
}
