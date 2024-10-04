use cosdata::storage::{
    bench_common,
    sparse_ann_query::{SparseAnnQuery, SparseAnnResult},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use rand::Rng;

const NUM_OF_VECTORS: usize = 1000; // Each of these will be used to create 100 more perturbed vectors
const NUM_OF_DIMENSIONS: usize = 50000;

fn sparse_ann_query_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Ann Query Benchmark");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::new(30, 0)); //Give enough time to measure
    let mut rng = rand::thread_rng();
    println!("Creating Inverted Index and query vector.. ");

    let (inverted_index, mut query_vector) =
        bench_common::create_inverted_index_and_query_vector(NUM_OF_DIMENSIONS, NUM_OF_VECTORS + 1);

    // Petrubing the query vector.
    let mut new_entries = Vec::with_capacity(query_vector.entries.len());
    for (dim, val) in &query_vector.entries {
        let perturbation = rng.gen_range(-0.5..=0.5);
        let new_val = (val + perturbation).clamp(0.0, 5.0);
        new_entries.push((*dim, new_val));
    }
    query_vector.entries = new_entries;

    let sparse_ann_query = SparseAnnQuery::new(query_vector);

    println!(
        "Starting benchmark.. for Vector count {:?} and dimension {:?}",
        NUM_OF_VECTORS * 100,
        NUM_OF_DIMENSIONS
    );
    // Benchmarking Sparse_Ann_Query_Concurrent
    group.bench_function(
        BenchmarkId::new(
            "Sparse_Ann_Query_Sequential",
            format!(
                "Total vectors = {} and dimensions = {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS,
            ),
        ),
        |b| {
            b.iter(|| {
                let _res: Vec<SparseAnnResult> =
                    sparse_ann_query.sequential_search(&inverted_index);
            });
        },
    );
}

criterion_group!(benches, sparse_ann_query_benchmark);
criterion_main!(benches);
