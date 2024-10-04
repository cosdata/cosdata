use cosdata::storage::bench_common;
use cosdata::storage::sparse_ann_query::SparseAnnQuery;
use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkId, Criterion};
use std::time::{Duration, Instant};

const NUM_OF_VECTORS: usize = 1000; // Each of these will be used to create 100 more perturbed vectors
const NUM_OF_DIMENSIONS: usize = 50000;
const NUM_OF_QUERY_VEC: usize = 200;

fn sparse_ann_query_rps_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Ann Query RequestPerSecond Validation Benchmark");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::new(5, 0)); //Give enough time to measure

    let (inverted_index, _query_vector) =
        bench_common::create_inverted_index_and_query_vector(NUM_OF_DIMENSIONS, NUM_OF_VECTORS + 1);

    //Generating 100 query vectors for checking successful result and time.
    let query_vecs =
        bench_common::generate_random_sparse_vectors(NUM_OF_QUERY_VEC, NUM_OF_DIMENSIONS);
    let mut sparse_ann_query_vec = Vec::new();
    for v in query_vecs {
        sparse_ann_query_vec.push(SparseAnnQuery::new(v));
    }

    group.bench_function(
        BenchmarkId::new(
            "Sparse_Ann_Query_RequestPerSecond_Validation",
            format!(
                "Inverted Index with Total vectors = {} and dimensions = {}. Number of query vectors : {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS,
                NUM_OF_QUERY_VEC
            ),
        ),
        |b| {
            b.iter(|| {
                let now = Instant::now();
                for (index,query) in &mut sparse_ann_query_vec.iter().enumerate(){
                    let _res = query.sequential_search(&inverted_index);
                    if now.elapsed() > Duration::from_millis(200){
                        println!("Number of queries successfully implemented within 200ms: {:?}",index);
                        break;
                    }
                }
            });
        },
    );
}

criterion_group!(benches, sparse_ann_query_rps_benchmark);
criterion_main!(benches);
