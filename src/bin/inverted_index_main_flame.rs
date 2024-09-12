use std::collections::BTreeSet;

use cosdata::storage::inverted_index::InvertedIndex;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

// Function to generate multiple random sparse vectors
fn generate_random_sparse_vectors(num_records: usize, dimensions: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(2024);
    let mut records: Vec<Vec<f32>> = Vec::with_capacity(num_records);

    for _ in 0..num_records {
        let mut record = vec![0.0; dimensions];

        // Calculate the number of non-zero elements (5% to 10% of dimensions)
        let num_nonzero = rng
            .gen_range((dimensions as f32 * 0.05) as usize..=(dimensions as f32 * 0.10) as usize);

        // BTreeSet is used to store unique indices of nonzero values in sorted order
        let mut unique_indices = BTreeSet::new();
        while unique_indices.len() < num_nonzero {
            let index = rng.gen_range(0..dimensions);
            unique_indices.insert(index);
        }

        // Generate random values for the nonzero indices
        for index in unique_indices {
            record[index] = rng.gen();
        }

        records.push(record);
    }

    records
}

fn main() {
    let dimensions = 1000; // Increased dimensions
    let num_vectors = 10_000;

    // Generate random sparse vectors
    let records = generate_random_sparse_vectors(num_vectors, dimensions);

    let inverted_index: InvertedIndex<f32> = InvertedIndex::new();
    records.par_iter().enumerate().for_each(|(id, record)| {
        let _ = inverted_index.add_sparse_vector(record.to_vec(), id as u32);
    });
}
