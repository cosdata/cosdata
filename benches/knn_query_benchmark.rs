use cosdata::models::types::SparseVector;
use cosdata::storage::{
    inverted_index_old::InvertedIndex,
    knn_query_old::{KNNQuery, KNNResult},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeSet;

const NUM_OF_VECTORS: usize = 2000; // Each of these will be used to create 100 more perturbed vectors
const NUM_OF_DIMENSIONS: usize = 50000;

// Function to generate multiple random sparse vectors
fn generate_random_sparse_vectors(num_records: usize, dimensions: usize) -> Vec<SparseVector> {
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
fn create_inverted_index_and_query_vector(
    num_dimensions: usize,
    num_vectors: usize,
) -> (InvertedIndex<f32>, SparseVector) {
    let inverted_index = InvertedIndex::new();

    let mut original_vectors: Vec<SparseVector> =
        generate_random_sparse_vectors(num_vectors as usize, num_dimensions as usize);
    let query_vector = original_vectors.pop().unwrap();

    let mut final_vectors = Vec::new(); // Final vectors generated after perturbation
    let mut current_id = NUM_OF_VECTORS + 1; // To ensure unique IDs

    for vector in original_vectors {
        // We create 100 new sparse vecs from each of original [NUM_OF_VECTORS] => NUM_OF_VECTORS * 100 = final_vectors
        let mut new_vectors = perturb_vector(&vector, 0.5, current_id);
        final_vectors.append(&mut new_vectors);
        current_id += 100; // Move to the next block of 100 IDs
    }

    for vector in final_vectors {
        inverted_index
            .add_sparse_vector(vector)
            .unwrap_or_else(|e| println!("Error : {:?}", e));
    }

    (inverted_index, query_vector)
}

fn perturb_vector(
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

fn knn_query_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn query benchmark");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::new(60, 0)); //Give enough time to collect samples
    let mut rng = rand::thread_rng();

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

    let knn_query = KNNQuery::new(query_vector);

    // Benchmarking Knn_Query_Concurrent
    group.bench_function(
        BenchmarkId::new(
            "Knn_Query_Concurrent",
            format!(
                "Total vectors = {} and dimensions = {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS,
            ),
        ),
        |b| {
            b.iter(|| {
                let _res: Vec<KNNResult> = knn_query.concurrent_search(&inverted_index);
            });
        },
    );

    // Benchmarking Knn_Query_Sequential
    group.bench_function(
        BenchmarkId::new(
            "Knn_Query_Sequential",
            format!(
                "Total vectors = {} and dimensions = {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS,
            ),
        ),
        |b| {
            b.iter(|| {
                let _res = knn_query.sequential_search(&inverted_index);
            });
        },
    );
}

criterion_group!(benches, knn_query_benchmark);
criterion_main!(benches);
