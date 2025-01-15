use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

use cosdata::models::types::SparseVector;
use cosdata::storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnBasic;
use cosdata::storage::sparse_ann_query_basic::{SparseAnnQueryBasic, SparseAnnResult};

const NUM_OF_VECTORS: usize = 1000000; // Adjust as needed
const NUM_OF_DIMENSIONS: u32 = 1024; // Adjust as needed
const K: usize = 5; // Number of nearest neighbors

fn bruteforce_vs_ann_benchmark(c: &mut Criterion) {
    // Create inverted index and generate test vectors
    let (inverted_index, test_vectors) =
        create_inverted_index_and_test_vectors(NUM_OF_DIMENSIONS, NUM_OF_VECTORS);

    // Choose a random query vector
    let mut rng = rand::thread_rng();
    let query_vector = test_vectors[rng.gen_range(0..test_vectors.len())].clone();

    // Perform ANN search using the inverted index
    let ann_query = SparseAnnQueryBasic::new(query_vector.clone());
    let ann_results = ann_query.sequential_search(&inverted_index);

    // Perform exhaustive brute-force search
    let brute_force_results = brute_force_search(&query_vector, &test_vectors);

    // Compare the results
    compare_results(&ann_results, &brute_force_results);

    // Benchmarking
    let mut group = c.benchmark_group("Brute Force vs ANN Benchmark");
    group.bench_function("ANN Search", |b| {
        b.iter(|| {
            ann_query.sequential_search(&inverted_index);
        });
    });
    group.bench_function("Brute Force Search", |b| {
        b.iter(|| {
            brute_force_search(&query_vector, &test_vectors);
        });
    });
    group.finish();
}

// Function to create inverted index and test vectors
fn create_inverted_index_and_test_vectors(
    num_dimensions: u32,
    num_vectors: usize,
) -> (InvertedIndexSparseAnnBasic, Vec<SparseVector>) {
    let inverted_index = InvertedIndexSparseAnnBasic::new();
    let mut vectors = Vec::with_capacity(num_vectors);

    for vector_id in 0..num_vectors {
        let sparse_vector = generate_random_sparse_vector(vector_id as u32, num_dimensions);
        inverted_index
            .add_sparse_vector(sparse_vector.clone())
            .expect("Unable to add sparse vector");
        vectors.push(sparse_vector);
    }

    (inverted_index, vectors)
}

// Function to generate a random sparse vector
fn generate_random_sparse_vector(vector_id: u32, num_dimensions: u32) -> SparseVector {
    let mut rng = rand::thread_rng();
    let num_nonzero = rng.gen_range(1..num_dimensions / 10); // Adjust sparsity
    let mut entries = Vec::with_capacity(num_nonzero as usize);

    for _ in 0..num_nonzero {
        let dim = rng.gen_range(0..num_dimensions);
        let val = rng.gen_range(-1.0..1.0);
        entries.push((dim, val));
    }

    SparseVector { vector_id, entries }
}

// Function to perform brute-force search
fn brute_force_search(
    query_vector: &SparseVector,
    vectors: &[SparseVector],
) -> Vec<SparseAnnResult> {
    let mut results = Vec::new();

    for vector in vectors {
        let similarity = compute_similarity(&query_vector.entries, &vector.entries);
        results.push(SparseAnnResult {
            vector_id: vector.vector_id,
            similarity: (similarity * 1000.0) as u32, // Scaling for comparison
        });
    }

    // Get top K results
    results.sort_by(|a, b| b.similarity.cmp(&a.similarity));
    results.truncate(K);

    results
}

// Function to compute similarity (dot product)
fn compute_similarity(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let mut sim = 0.0;
    let mut a_iter = a.iter();
    let mut b_iter = b.iter();
    let (mut a_item, mut b_item) = (a_iter.next(), b_iter.next());

    while let (Some(&(a_dim, a_val)), Some(&(b_dim, b_val))) = (a_item, b_item) {
        match a_dim.cmp(&b_dim) {
            std::cmp::Ordering::Equal => {
                sim += a_val * b_val;
                a_item = a_iter.next();
                b_item = b_iter.next();
            }
            std::cmp::Ordering::Less => {
                a_item = a_iter.next();
            }
            std::cmp::Ordering::Greater => {
                b_item = b_iter.next();
            }
        }
    }

    sim
}

// Function to compare ANN results to brute-force results
fn compare_results(ann_results: &[SparseAnnResult], bf_results: &[SparseAnnResult]) {
    for (ann_result, bf_result) in ann_results.iter().zip(bf_results.iter()) {
        assert_eq!(ann_result, bf_result);
    }
    println!("ANN results match brute-force results.");
}
criterion_group!(benches, bruteforce_vs_ann_benchmark);
criterion_main!(benches);
