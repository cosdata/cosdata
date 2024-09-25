use cosdata::models::types::SparseVector;
use cosdata::storage::{
    inverted_index::InvertedIndex,
    knn_query::{KNNQuery, KNNResult},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use rand::Rng;

const NUM_OF_VECTORS: u32 = 10000;
const NUM_OF_DIMENSIONS: u32 = 200;
const RANGE_OF_VALUES: (f32, f32) = (0.0, 3.0);

fn create_random_sparse_vector(
    vector_id: u32,
    num_dimensions: u32,
    value_range: (f32, f32),
) -> SparseVector {
    let mut rng = rand::thread_rng();
    let no_of_entries = rng.gen_range(0..num_dimensions); // Randomly generating number of entries per vector
    let mut entries = Vec::new();
    // (dim,value),(dim,value),... no_of_entries
    for _ in 0..no_of_entries {
        let dim: u32 = rng.gen_range(0..num_dimensions); // Randomly generating dimensions in entries
        let value: f32 = rng.gen_range(value_range.0..value_range.1); // Randomly generating values in entries
        entries.push((dim, value));
    }
    SparseVector::new(vector_id, entries)
}

fn create_inverted_index(
    num_dimensions: u32,
    num_vectors: u32,
    value_range: (f32, f32),
) -> InvertedIndex<f32> {
    let inverted_index = InvertedIndex::new();

    for vector_id in 1..num_vectors {
        let vec = create_random_sparse_vector(vector_id, num_dimensions, value_range);
        inverted_index
            .add_sparse_vector(vec)
            .unwrap_or_else(|e| println!("Error : {:?}", e));
    }

    inverted_index
}

fn knn_query_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn query benchmark");
    group.sample_size(10);

    let test_sparse_vector = create_random_sparse_vector(0, NUM_OF_DIMENSIONS, RANGE_OF_VALUES);
    let test_index = create_inverted_index(NUM_OF_DIMENSIONS, NUM_OF_VECTORS, RANGE_OF_VALUES);
    let knn_query = KNNQuery::new(test_sparse_vector);

    // ALl benchmarks have
    // Benchmarking Knn_query_sequential with index and concurrency
    group.bench_function(BenchmarkId::new("Knn_query_concurrent", 1000), |b| {
        b.iter(|| {
            let res: Vec<KNNResult> = knn_query.concurrent_search(&test_index);
        });
    });

    // Benchmarking Knn_query_sequential with index and no concurrency
    group.bench_function(BenchmarkId::new("Knn_query_sequential", 1000), |b| {
        b.iter(|| {
            let res = knn_query.sequential_search(&test_index);
        });
    });

    // Benchmarking Knn_query_brute with no index and no concurrency
    group.bench_function(BenchmarkId::new("Knn_query_brute", 1000), |b| {
        b.iter(|| {
            let mut sparse_vecs = Vec::new();
            for i in 0..NUM_OF_VECTORS {
                let svec = create_random_sparse_vector(i, NUM_OF_DIMENSIONS, RANGE_OF_VALUES);
                sparse_vecs.push(svec);
            }
            let res = knn_query.brute_search(sparse_vecs);
        });
    });
}

criterion_group!(benches, knn_query_benchmark);
criterion_main!(benches);
