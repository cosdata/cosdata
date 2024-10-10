use cosdata::models::types::SparseVector;
use cosdata::storage::bench_common;
use cosdata::storage::sparse_ann_query::SparseAnnQuery;
use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkId, Criterion};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const NUM_OF_VECTORS: usize = 1000; // Each of these will be used to create 100 more perturbed vectors
const NUM_OF_DIMENSIONS: usize = 50000;
const NUM_OF_QUERY_VEC: usize = 1000;
#[derive(Debug)]
pub struct TimeStats {
    batch_length: u32,
    time_taken: Duration,
}

impl TimeStats {
    pub fn new(batch_length: u32, time_taken: Duration) -> TimeStats {
        TimeStats {
            batch_length,
            time_taken,
        }
    }
}

fn sparse_ann_query_rps_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Ann Query RequestPerSecond Validation Benchmark");
    group
        .sample_size(10)
        .measurement_time(std::time::Duration::new(10, 0)); //Give enough time to measure

    let (inverted_index, _query_vector) =
        bench_common::create_inverted_index_and_query_vector(NUM_OF_DIMENSIONS, NUM_OF_VECTORS + 1);

    //Generating 1000 query vectors for checking successful result and time.
    let query_vecs: Vec<SparseVector> =
        bench_common::generate_random_sparse_vectors(NUM_OF_QUERY_VEC, NUM_OF_DIMENSIONS);

    let mut sparse_ann_query_vec = Vec::new();
    for vex in query_vecs {
        sparse_ann_query_vec.push(SparseAnnQuery::new(vex.clone()));
    }

    let mut batch_query: Vec<Vec<SparseAnnQuery>> = Vec::new(); //Batches of [100,200,300,400] from 1000 query_vecs

    for i in 1..5 {
        if i * 100 > sparse_ann_query_vec.len() {
            break;
        }
        let query_rem = sparse_ann_query_vec.split_off(i * 100);
        println!(
            "Just inserting a batch of sparse_ann_query_vec {:?}",
            i * 100
        );
        batch_query.push(sparse_ann_query_vec);
        sparse_ann_query_vec = query_rem;
    }

    println!("Number of batches in batch_query {:?}", batch_query.len());
    group.bench_function(
        BenchmarkId::new(
            "Sparse_Ann_Query_RequestPerSecond_Validation",
            format!(
                "Inverted Index with Total vectors = {} and dimensions = {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS
            ),
        ),
        |b| {
            b.iter(|| {
                for q_vec in &batch_query {
                    let now = Instant::now();
                    let batch_length = q_vec.len();
                    let mut queries_successful = 0;
                    for (index, query) in &mut q_vec.iter().enumerate() {
                        let _res = query.sequential_search(&inverted_index);
                        if now.elapsed() > Duration::from_millis(200) {
                            queries_successful = index;
                            break;
                        } else if index == batch_length - 1 {
                            queries_successful = batch_length;
                        }
                    }
                    let time_elapsed = now.elapsed();
                    println!(
                        "Queries successfully run {:?} in batch length {:?} in time {:?}",
                        queries_successful, batch_length, time_elapsed
                    );
                    if (queries_successful as f32 / batch_length as f32) < 0.5 {
                        // Breaking batch loop if less than 50% are successful
                        break;
                    }
                }
            });
        },
    );

    let mut time_distribution: Vec<TimeStats> = Vec::new();

    group.bench_function(
        BenchmarkId::new(
            "Sparse_Ann_Query_Time_taken_assessment",
            format!(
                "Inverted Index with Total vectors = {} and dimensions = {}",
                NUM_OF_VECTORS * 100,
                NUM_OF_DIMENSIONS
            ),
        ),
        |b| {
            b.iter(|| {
                for q_vec in &batch_query {
                    let now = Instant::now();
                    for x in q_vec.iter() {
                        let _res = x.sequential_search(&inverted_index);
                    }
                    let stat = TimeStats::new(q_vec.len() as u32, now.elapsed());
                    time_distribution.push(stat);
                }
            })
        },
    );

    let mut time_batches: HashMap<u32, Vec<Duration>> = HashMap::new();

    time_distribution.iter().for_each(|x| {
        time_batches
            .entry(x.batch_length)
            .or_insert_with(Vec::new)
            .push(x.time_taken);
    });

    for (batch_size, vec_dur) in time_batches {
        println!(
            "\n For batch size {:?}, \nThe mean time is {:?}, \n variance is {:?}ms, \n Standard deviation is {:?}ms, \n",batch_size,
            bench_common::mean(&vec_dur), format!("{:.5}", bench_common::variance(&vec_dur)),bench_common::standard_deviation(&vec_dur),
        );
    }
}

criterion_group!(benches, sparse_ann_query_rps_benchmark);
criterion_main!(benches);
