mod vector_store;
mod api_service;
mod web_server;
mod models;

use async_std::task;
use lazy_static::lazy_static;
use rand::Rng;
use std::f64;
use std::sync::{Arc, Mutex};

lazy_static! {
    static ref random_numbers_a: Vec<f32> = generate_random_vector();
    static ref random_numbers_b: Vec<f32> = generate_random_vector();
}
use vector_store::{
    compute_cosine_similarity, cosine_coalesce, cosine_sim_unsigned, cosine_similarity, floats_to_bits, quantize, VectorEmbedding, VectorStore, VectorTreeNode
};

// Function to generate a vector of 1024 f32 random numbers in the range -1.0 to 1.0
fn generate_random_vector() -> Vec<f32> {
    // Create a mutable vector to store the random numbers
    let mut rng = rand::thread_rng();
    let mut numbers = Vec::with_capacity(1024);

    // Generate 1024 random numbers and push them to the vector
    for _ in 0..1024 {
        let num: f32 = rng.gen_range(-1.0..1.0);
        numbers.push(num);
    }

    // Return the vector of random numbers
    numbers
}

fn run_quantized_cs() -> f64 {
    let quantized_values_x = quantize(&random_numbers_a);
    let quantized_values_y = quantize(&random_numbers_b);

    // Call cosine_coalesce function with the pair of quantized values
    let similarity = cosine_coalesce(&quantized_values_x, &quantized_values_y);
    // println!("{:.8}", similarity);
    return similarity;
}

fn run_cs() -> f32 {
    let similarity = cosine_similarity(&random_numbers_a, &random_numbers_b);
    return similarity;
}

fn run_cs_new() -> f64 {
    let x = floats_to_bits(&random_numbers_a);
    let y = floats_to_bits(&random_numbers_b);

    let similarity = cosine_sim_unsigned(&x, &y);
    return similarity;
}

fn main() {
    async_std::task::block_on(async {
        let tasks = (0..(932500* 4))
            .map(|_| {
                let (sender, receiver) = async_channel::bounded(1);
                task::spawn(async move {
                    let similarity = run_cs_new();
                    sender
                        .send(similarity)
                        .await
                        .expect("Sending result failed");
                });
                receiver
            })
            .collect::<Vec<_>>();

        for task in tasks {
            task.recv().await.expect("Receiving result failed");
            // println!("{:.8}", task.recv().await.expect("Receiving result failed"));
        }
    });
}
