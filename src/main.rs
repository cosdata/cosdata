mod vector_store;
use rand::Rng;
use std::f64;
use async_std::task;
use std::sync::{Arc, Mutex};
use async_channel::Sender;

use vector_store::{
    compute_cosine_similarity, cosine_coalesce, cosine_similarity, quantize, VectorEmbedding,
    VectorStore, VectorTreeNode,
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

fn run() -> f64 {
    let random_numbers_a = generate_random_vector();
    let random_numbers_b = generate_random_vector();

    let quantized_values_x = quantize(&random_numbers_a);
    let quantized_values_y = quantize(&random_numbers_b);

    // Call cosine_coalesce function with the pair of quantized values
    let similarity = cosine_coalesce(&quantized_values_x, &quantized_values_y);
    // println!("{:.8}", similarity);
    return similarity;
}
fn main() {
    // Example bit sequence
    let bits1 = [1, 0, 1, 1, 0, 1, 1, 1]; // Represents the binary value 101101
    let bits2 = [1, 0, 0, 1, 0, 0, 0, 1]; // Represents the binary value 101101

    // Compute the cosine similarity
    let result = cosine_similarity(&bits1, &bits2);
    println!("Cosine similarity result: {}", result);

    // Compute the CosResult struct
    let cos_result = compute_cosine_similarity(&bits1, &bits2, 8);
    println!(
        "CosResult -> dotprod: {}, premagA: {}, premagB: {}",
        cos_result.dotprod, cos_result.premag_a, cos_result.premag_b
    );

    async_std::task::block_on(async {
        let tasks = (0..10000)
            .map(|_| {
                let (sender, receiver) = async_channel::bounded(1);
                task::spawn(async move {
                    let similarity = run();
                    sender
                        .send(similarity)
                        .await
                        .expect("Sending result failed");
                });
                receiver
            })
            .collect::<Vec<_>>();

        for task in tasks {
            println!("{:.8}", task.recv().await.expect("Receiving result failed"));
        }
    });
}
