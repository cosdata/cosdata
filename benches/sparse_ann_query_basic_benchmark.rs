use std::time::Instant;

use cosdata::{
    models::types::SparseVector,
    storage::{
        bench_common, inverted_index_sparse_ann_basic::InvertedIndexSparseAnnBasic,
        sparse_ann_query_basic::SparseAnnQueryBasic,
    },
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use rand::Rng;

const NUM_OF_VECTORS: usize = 3000; // Each of these will be used to create 100 more perturbed vectors
const NUM_OF_DIMENSIONS: usize = 50000;

// Returns inverted_index and query vector
pub fn create_inverted_index_and_query_vector(
    num_dimensions: usize,
    num_vectors: usize,
) -> (InvertedIndexSparseAnnBasic, SparseVector) {
    let nowx = Instant::now();
    let inverted_index = InvertedIndexSparseAnnBasic::new();

    let mut original_vectors: Vec<SparseVector> =
        bench_common::generate_random_sparse_vectors(num_vectors as usize, num_dimensions as usize);
    let query_vector = original_vectors.pop().unwrap();

    let mut final_vectors = Vec::new(); // Final vectors generated after perturbation
    let mut current_id = num_vectors + 1; // To ensure unique IDs

    for vector in original_vectors {
        // We create 100 new sparse vecs from each of original [NUM_OF_VECTORS] => NUM_OF_VECTORS * 100 = final_vectors
        let mut new_vectors = bench_common::perturb_vector(&vector, 0.5, current_id);
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

fn sparse_ann_query_basic_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Ann Query Basic Benchmark");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::new(30, 0)); //Give enough time to measure
    let mut rng = rand::thread_rng();
    println!("Creating Inverted Index and query vector.. ");

    let (inverted_index, mut query_vector) =
        create_inverted_index_and_query_vector(NUM_OF_DIMENSIONS, NUM_OF_VECTORS + 1);

    // Petrubing the query vector.
    let mut new_entries = Vec::with_capacity(query_vector.entries.len());
    for (dim, val) in &query_vector.entries {
        let perturbation = rng.gen_range(-0.5..=0.5);
        let new_val = (val + perturbation).clamp(0.0, 5.0);
        new_entries.push((*dim, new_val));
    }
    query_vector.entries = new_entries;

    let sparse_ann_query_basic = SparseAnnQueryBasic::new(query_vector);

    println!(
        "Starting benchmark.. for Vector count {:?} and dimension {:?}",
        NUM_OF_VECTORS * 100,
        NUM_OF_DIMENSIONS
    );
    // Benchmarking Sparse_Ann_Query_Sequential
    group.bench_function(
        BenchmarkId::new(
            "Sparse_Ann_Query_Basic_Sequential",
            format!(
                "Total vectors = {} and dimensions = {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS,
            ),
        ),
        |b| {
            b.iter(|| {
                let _res = sparse_ann_query_basic.sequential_search(&inverted_index);
            });
        },
    );
}

criterion_group!(benches, sparse_ann_query_basic_benchmark);
criterion_main!(benches);
