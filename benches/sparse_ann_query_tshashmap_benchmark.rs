use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use cosdata::{
    models::types::SparseVector,
    storage::{
        bench_common, inverted_index_sparse_ann_basic::InvertedIndexSparseAnnBasicTSHashmap,
        sparse_ann_query_basic::SparseAnnQueryBasic,
    },
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

const NUM_OF_VECTORS: usize = 100000; // Each of these will be used to create 100 more perturbed vectors
const NUM_OF_DIMENSIONS: usize = 50000;

// Returns inverted_index and query vector
pub fn create_inverted_index_and_query_vector(
    num_dimensions: usize,
    num_vectors: usize,
) -> (InvertedIndexSparseAnnBasicTSHashmap, SparseVector) {
    let inverted_index = InvertedIndexSparseAnnBasicTSHashmap::new();

    let mut original_vectors: Vec<SparseVector> =
        bench_common::generate_random_sparse_vectors(num_vectors as usize, num_dimensions as usize);
    let query_vector = original_vectors.pop().unwrap();

    let current_id = Arc::new(AtomicUsize::new(num_vectors + 1)); // To ensure unique IDs
    let now = Instant::now();

    let chunks = original_vectors.chunks(num_vectors / 10);
    for (i, chunk) in chunks.enumerate() {
        println!("Starting with chunk : {i}");
        chunk.par_iter().for_each(|vector| {
            let start_id = current_id.fetch_add(100, Ordering::SeqCst); // Move to the next block of 100 IDs

            // We create 100 new sparse vecs from each of original [NUM_OF_VECTORS] => NUM_OF_VECTORS * 100 = final_vectors
            let new_vectors = bench_common::perturb_vector(&vector, 0.5, start_id);
            for x in new_vectors {
                if x.vector_id % 10000 == 0 {
                    println!("Just added vectors : {:?}", x.vector_id);
                    println!("Time elapsed : {:?} secs", now.elapsed().as_secs_f32());
                }
                inverted_index
                    .add_sparse_vector(x)
                    .unwrap_or_else(|e| println!("Error : {:?}", e));
            }
        });
        println!(
            "Finished adding chunk {i} in time {:?}",
            now.elapsed().as_secs_f32()
        );
    }
    // for vector in original_vectors {
    //     // We create 100 new sparse vecs from each of original [NUM_OF_VECTORS] => NUM_OF_VECTORS * 100 = final_vectors
    //     let new_vectors = bench_common::perturb_vector(&vector, 0.5, current_id);
    //     for x in new_vectors {
    //         if x.vector_id % 10000 == 0 {
    //             println!("Just added vectors : {:?}", x.vector_id);
    //             println!("Time elapsed : {:?} secs", now.elapsed().as_secs_f32());
    //         }
    //         inverted_index
    //             .add_sparse_vector(x)
    //             .unwrap_or_else(|e| println!("Error : {:?}", e));
    //     }
    //     current_id += 100; // Move to the next block of 100 IDs
    // }

    println!(
        "Finished generating {:?} vectors and adding them to inverted index in time {:?}",
        NUM_OF_VECTORS * 100,
        now.elapsed()
    );
    println!(
        "Time taken to insert all vectors : {:?} secs",
        now.elapsed().as_secs_f32()
    );

    (inverted_index, query_vector)
}

fn sparse_ann_query_tshashmap_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Ann Query Basic Benchmark");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::new(30, 0)); //Give enough time to measure
    let mut rng = rand::thread_rng();
    println!("Creating Inverted Index [InvertedIndexSparseAnnBasicTSHashmap] and query vector.. ");

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
            "Sparse_Ann_Query_Basic_TSHashmap_Sequential",
            format!(
                "Total vectors = {} and dimensions = {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS,
            ),
        ),
        |b| {
            b.iter(|| {
                let _res = sparse_ann_query_basic.sequential_search_tshashmap(&inverted_index);
            });
        },
    );
}

criterion_group!(benches, sparse_ann_query_tshashmap_benchmark);
criterion_main!(benches);
