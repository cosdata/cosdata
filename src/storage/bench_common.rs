use super::inverted_index_sparse_ann::InvertedIndexSparseAnn;
use crate::models::types::SparseVector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

/// This bench_common file contains all relevant functions required for benchmarking knn and sparse_ann queries
/// Import these functions in benchmarking, for easy use.

pub fn mean(data: &Vec<Duration>) -> Duration {
    let sum: Duration = data.iter().sum();
    sum / data.len() as u32
}

pub fn variance(data: &Vec<Duration>) -> f64 {
    let data_mean = mean(data).as_secs_f64();
    data.iter()
        .map(|value| {
            let diff = data_mean - value.as_secs_f64();
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64
}

pub fn standard_deviation(data: &Vec<Duration>) -> f32 {
    variance(data).sqrt() as f32
}

// Function to generate multiple random sparse vectors
pub fn generate_random_sparse_vectors(num_records: usize, dimensions: usize) -> Vec<SparseVector> {
    let mut rng = StdRng::seed_from_u64(2024);
    let mut records: Vec<SparseVector> = Vec::with_capacity(num_records);

    for vector_id in 0..num_records {
        // Calculate the number of non-zero elements (0.5% to 0.6% of dimensions)
        let num_nonzero = rng
            .gen_range((dimensions as f32 * 0.005) as usize..=(dimensions as f32 * 0.006) as usize);
        let mut entries: Vec<(u32, f32)> = Vec::with_capacity(num_nonzero);

        // BTreeSet is used to store unique indices of nonzero values in sorted order
        let mut unique_indices = BTreeSet::new();
        while unique_indices.len() < num_nonzero {
            let index = rng.gen_range(0..dimensions);
            unique_indices.insert(index);
        }

        // Generate random values for the nonzero indices
        for dim_index in unique_indices {
            let value = rng.gen();
            entries.push((dim_index as u32, value));
        }

        records.push(SparseVector::new(vector_id as u32, entries));
    }

    records
}

// Returns inverted_index and query vector
pub fn create_inverted_index_and_query_vector(
    num_dimensions: usize,
    num_vectors: usize,
) -> (InvertedIndexSparseAnn, SparseVector) {
    let nowx = Instant::now();
    let inverted_index = InvertedIndexSparseAnn::new();

    let mut original_vectors: Vec<SparseVector> =
        generate_random_sparse_vectors(num_vectors as usize, num_dimensions as usize);
    let query_vector = original_vectors.pop().unwrap();

    let mut final_vectors = Vec::new(); // Final vectors generated after perturbation
    let mut current_id = num_vectors + 1; // To ensure unique IDs

    for vector in original_vectors {
        // We create 100 new sparse vecs from each of original [NUM_OF_VECTORS] => NUM_OF_VECTORS * 100 = final_vectors
        let mut new_vectors = perturb_vector(&vector, 0.5, current_id);
        final_vectors.append(&mut new_vectors);
        current_id += 100; // Move to the next block of 100 IDs
    }

    println!(
        "Finished generating vectors in time : {:?}, Next step adding them to inverted index...",
        nowx.elapsed()
    );
    let now = Instant::now();
    for vector in final_vectors {
        if vector.vector_id % 10000 == 0 {
            println!("Just added vectors : {:?}", vector.vector_id);
            println!("Time elapsed : {:?} secs", now.elapsed().as_secs_f32());
        }
        inverted_index
            .add_sparse_vector(vector)
            .unwrap_or_else(|e| println!("Error : {:?}", e));
    }
    println!(
        "Time taken to insert all vectors : {:?} secs",
        now.elapsed().as_secs_f32()
    );

    (inverted_index, query_vector)
}

pub fn perturb_vector(
    vector: &SparseVector,
    perturbation_degree: f32,
    start_id: usize,
) -> Vec<SparseVector> {
    let mut rng = StdRng::seed_from_u64(2024);
    let mut new_vectors = Vec::new();

    for i in 0..100 {
        let new_vector_id = start_id + i; // Generating unique ID for each new vector
        let mut new_entries = Vec::with_capacity(vector.entries.len());

        for &(dim, val) in &vector.entries {
            let perturbation = rng.gen_range(-perturbation_degree..=perturbation_degree);
            let new_val = (val + perturbation).clamp(0.0, 5.0);
            new_entries.push((dim, new_val));
        }

        new_vectors.push(SparseVector {
            vector_id: new_vector_id as u32,
            entries: new_entries,
        });
    }

    new_vectors // Sending 100 new vectors from each original sparse_vector received.
}
