mod api_service;
mod models;
mod vector_store;
mod web_server;
use async_std::task;
use lazy_static::lazy_static;
use rand::Rng;
use std::f64;
use std::sync::{Arc, Mutex};
use web_server::run_actix_server;

lazy_static! {
    static ref RANDOM_NUMBERS_A: Vec<f32> = generate_random_vector();
    static ref RANDOM_NUMBERS_B: Vec<f32> = generate_random_vector();
    static ref QUANTIZED_VALUES_A: Vec<Vec<u8>> = quantize_to_u8_bits(&RANDOM_NUMBERS_A);
    static ref QUANTIZED_VALUES_B: Vec<Vec<u8>> = quantize_to_u8_bits(&RANDOM_NUMBERS_B);
    static ref MPQ_A: (f64, Vec<u32>) = get_magnitude_plus_quantized_vec(QUANTIZED_VALUES_A.to_vec());
    static ref MPQ_B: (f64, Vec<u32>) = get_magnitude_plus_quantized_vec(QUANTIZED_VALUES_B.to_vec());
}

use crate::models::common::*;

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

fn run_cs() -> f32 {
    let similarity = cosine_similarity(&RANDOM_NUMBERS_A, &RANDOM_NUMBERS_B);
    return similarity;
}



fn main() {
    async_std::task::block_on(async {
        let tasks = (0..(4))
            .map(|_| {
                let (sender, receiver) = async_channel::bounded(1);
                task::spawn(async move {
                    let similarity = run_cs();
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
    run_actix_server();
}
