mod vector_store;

use std::f64;
use vector_store::{cosine_similarity, compute_cosine_similarity, VectorStore, VectorEmbedding, VectorTreeNode}; 

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
}
