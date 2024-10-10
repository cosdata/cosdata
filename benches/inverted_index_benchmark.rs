use std::collections::BTreeSet;

use cosdata::{models::types::SparseVector, storage::inverted_index_old::InvertedIndex};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

// Function to generate multiple random sparse vectors
fn generate_random_sparse_vectors(num_records: usize, dimensions: usize) -> Vec<SparseVector> {
    let mut rng = StdRng::seed_from_u64(2024);
    let mut records: Vec<SparseVector> = Vec::with_capacity(num_records);

    for vector_id in 0..num_records {
        // Calculate the number of non-zero elements (5% to 10% of dimensions)
        let num_nonzero = rng
            .gen_range((dimensions as f32 * 0.05) as usize..=(dimensions as f32 * 0.10) as usize);
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
                    for record in records.iter() {
                        for entry in record.entries.iter() {
                            let _ = inverted_index.insert(entry.0, entry.1, record.vector_id);
                        }
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
                    records.par_iter().for_each(|record| {
                        let _ = inverted_index.add_sparse_vector(record.clone());
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
