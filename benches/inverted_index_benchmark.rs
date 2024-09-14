use std::collections::BTreeSet;

use cosdata::storage::inverted_index::InvertedIndex;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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

fn benchmark_sequential_inserts(c: &mut Criterion) {
    let dimensions = 1000; // Increased dimensions
    let vector_counts = [1000];

    let mut group = c.benchmark_group("Insert sparse vectors");
    group.sample_size(10);

    for &num_vectors in &vector_counts {
        // Generate random sparse vectors
        let records = generate_random_sparse_vectors(num_vectors, dimensions);

        // Sequential benchmark
        let seq_test_name = format!("Sequential {} vectors", num_vectors);
        group.bench_with_input(
            BenchmarkId::new(seq_test_name, num_vectors),
            &records,
            |b, records| {
                b.iter(|| {
                    let inverted_index: InvertedIndex<f32> = InvertedIndex::new();
                    for (id, record) in records.iter().enumerate() {
                        let _ = inverted_index.add_sparse_vector(record.to_vec(), id as u32);
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_parallel_inserts(c: &mut Criterion) {
    let dimensions = 1000; // Increased dimensions
    let vector_counts = [1000, 10000, 100_000];

    let mut group = c.benchmark_group("Insert sparse vectors");
    group.sample_size(10);

    for &num_vectors in &vector_counts {
        // Generate random sparse vectors
        let records = generate_random_sparse_vectors(num_vectors, dimensions);

        // Parallel benchmark
        let par_test_name = format!("Parallel {} vectors", num_vectors);
        group.bench_with_input(
            BenchmarkId::new(par_test_name, num_vectors),
            &records,
            |b, records| {
                b.iter(|| {
                    let inverted_index: InvertedIndex<f32> = InvertedIndex::new();
                    records.par_iter().enumerate().for_each(|(id, record)| {
                        let _ = inverted_index.add_sparse_vector(record.to_vec(), id as u32);
                    });
                });
            },
        );
    }
}

criterion_group!(
    benches,
    // benchmark_sequential_inserts,
    benchmark_parallel_inserts
);
criterion_main!(benches);
