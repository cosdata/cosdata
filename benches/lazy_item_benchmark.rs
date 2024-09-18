use cosdata::models::{
    lazy_load::{LazyItem, LazyItemMap, LazyItemSet, LazyItemVec},
    types::SparseVector,
    versioning::Hash,
};
use cosdata::models::identity_collections::IdentityMapKey;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeSet;

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

fn lazy_item_benchmark(c: &mut Criterion) {
    let dimensions = 1000; // Increased dimensions
    let sparse_vector_count = 10000;

    let mut group = c.benchmark_group("Lazy item");
    group.sample_size(10);

    // Generate random sparse vectors
    let records = generate_random_sparse_vectors(sparse_vector_count, dimensions);

    //Sequential inserting sparse_vectors to LazyItemVec
    group.bench_with_input(
        BenchmarkId::new("LazyItemVec", sparse_vector_count),
        &records,
        |b, records| {
            b.iter(|| {
                let lazy_item_vec: LazyItemVec<SparseVector> = LazyItemVec::new();
                for (index, sparse_vec) in records.iter().enumerate() {
                    let version_hash = Hash::from(index as u32);
                    let _ = lazy_item_vec.insert(index, LazyItem::new(version_hash, sparse_vec.clone()));
                }
            });
        },
    );

    //Sequential inserting sparse_vectors to LazyItemMap
    group.bench_with_input(
        BenchmarkId::new("LazyItemMap", sparse_vector_count),
        &records,
        |b, records| {
            b.iter(|| {
                let lazy_item_map: LazyItemMap<SparseVector> = LazyItemMap::new();
                for (index, sparse_vec) in records.iter().enumerate() {
                    let version_hash = Hash::from(index as u32);
                    let _ = lazy_item_map.insert(IdentityMapKey::String(index.to_string()), LazyItem::new(version_hash, sparse_vec.clone()));
                }
            });
        },
    );
}

criterion_group!(benches, lazy_item_benchmark);
criterion_main!(benches);
