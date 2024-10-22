// # Project & Partition comparison with brute force.
//
// Generates 50k base vectors and 100 perturbations for each vector (50k x 100 = 5M).
// Randomly picks 100 vectors from the base vectors and generates perturbation of each,
// and saves them as query vectors.
//
// A sample run with 10M vectors.
//
// ## Brute force
//
// 01. 73303:25 (0.82679725)
// 02. 73303:24 (0.82381266)
// 03. 73303:63 (0.82380205)
// 04. 73303:23 (0.8174631)
// 05. 73303:84 (0.8103554)
// 06. 73303:65 (0.8101841)
// 07. 73303:87 (0.80889463)
// 08. 73303:67 (0.8088855)
// 09. 73303:98 (0.8085643)
// 10. 73303:39 (0.80775493)
// 11. 73303:70 (0.80716753)
// 12. 73303:73 (0.80707437)
// 13. 73303:46 (0.8068478)
// 14. 73303:86 (0.80590945)
// 15. 73303:57 (0.80557406)
// 16. 73303:97 (0.80540997)
// 17. 73303:37 (0.8049393)
// 18. 73303:92 (0.8028671)
// 19. 73303:62 (0.8019657)
// 20. 73303:1  (0.8018392)
//
// ## Project & Partition
//
// 01. 73303:25 (0.82679725)
// 02. 73303:24 (0.82381266)
// 03. 73303:63 (0.82380205)
// 04. 73303:23 (0.8174631)
// 05. 73303:84 (0.8103554)
// 06. 73303:87 (0.80889463)
// 07. 73303:67 (0.8088855)
// 08. 73303:98 (0.8085643)
// 09. 73303:39 (0.80775493)
// 10. 73303:70 (0.80716753)
// 11. 73303:73 (0.80707437)
// 12. 73303:46 (0.8068478)
// 13. 73303:86 (0.80590945)
// 14. 73303:57 (0.80557406)
// 15. 73303:97 (0.80540997)
// 16. 73303:37 (0.8049393)
// 17. 73303:92 (0.8028671)
// 18. 73303:62 (0.8019657)
// 19. 73303:1  (0.8018392)
// 20. 73303:59 (0.800483)
//
// ## Performance
//
// Brute force:
//   - Query time: ~ 12s
//
// Project & Partition:
//   - Index creation time: ~ 92s
//   - Query time: ~ 469ms

use std::{
    cmp::Ordering,
    fs,
    path::Path,
    time::{Duration, Instant},
};

use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

#[derive(Clone)]
struct ProjectedValue {
    value: f32,
    weight: f32,
    iterations: u8,
}

fn is_sensitive_pair(x: f32, y: f32) -> bool {
    if x == 0.0 {
        y.abs() > 0.85
    } else {
        x.abs() < (0.11764705882352941) && (y).abs() > 0.85
    }
}

fn calculate_weight(iteration: u8) -> f32 {
    2.0_f32.powi(iteration as i32)
}

fn project_to_3d(x: f32, y: f32) -> f32 {
    2.0 * (y / x).atan() / std::f32::consts::PI
}

fn project_vector_to_x(v: &[f32], x: usize) -> Vec<ProjectedValue> {
    let mut projected = v
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                ProjectedValue {
                    value: chunk[0],
                    weight: if is_sensitive_pair(chunk[0], chunk[1]) {
                        calculate_weight(1)
                    } else {
                        0.0
                    },
                    iterations: 1,
                }
            } else {
                ProjectedValue {
                    value: chunk[0],
                    weight: 0.0,
                    iterations: 1,
                }
            }
        })
        .collect::<Vec<_>>();

    let mut iteration: u8 = 2;
    while projected.len() > x {
        if projected.len() / 2 < x {
            let pairs_to_project = projected.len() - x;
            let mut new_projected = Vec::new();

            for i in 0..pairs_to_project {
                let z = project_to_3d(projected[2 * i].value, projected[2 * i + 1].value);
                let inherited_weight = projected[2 * i].weight + projected[2 * i + 1].weight;
                let sensitivity_weight =
                    if is_sensitive_pair(projected[2 * i].value, projected[2 * i + 1].value) {
                        calculate_weight(iteration)
                    } else {
                        0.0
                    };
                let new_weight = inherited_weight + sensitivity_weight;

                new_projected.push(ProjectedValue {
                    value: z,
                    weight: new_weight,
                    iterations: iteration,
                });
            }

            new_projected.extend_from_slice(&projected[pairs_to_project * 2..]);
            projected = new_projected;
        } else {
            projected = projected
                .chunks(2)
                .map(|chunk| {
                    let z = project_to_3d(chunk[0].value, chunk[1].value);
                    let inherited_weight = chunk[0].weight + chunk[1].weight;
                    let sensitivity_weight = if is_sensitive_pair(chunk[0].value, chunk[1].value) {
                        calculate_weight(iteration)
                    } else {
                        0.0
                    };
                    let new_weight = inherited_weight + sensitivity_weight;

                    ProjectedValue {
                        value: z,
                        weight: new_weight,
                        iterations: iteration,
                    }
                })
                .collect();
        }
        iteration = iteration.saturating_add(1);
    }
    projected
}

fn make_index(v: &[ProjectedValue]) -> (u16, Vec<u16>) {
    let mut main_index = 0;
    let mut main_mask = 0;
    let mut sway_bits = Vec::new();

    let max_iterations = v.iter().map(|pv| pv.iterations).max().unwrap_or(0);
    let max_possible_weight: f32 = (1..=max_iterations).map(calculate_weight).sum();
    let threshold = max_possible_weight / 100.0;

    for (i, pv) in v.iter().enumerate() {
        if pv.value >= 0.0 {
            main_index |= 1 << i;
        }
        if pv.value >= 0.15 {
            main_mask |= 1 << i;
        } else if pv.weight > threshold {
            sway_bits.push(i as u16);
        }
    }

    let mut alt_indices = vec![main_mask];
    for i in 1..2u16.pow(sway_bits.len() as u32) {
        let mut alt_index = main_mask;
        for (j, &bit) in sway_bits.iter().enumerate() {
            if (i & (1 << j)) != 0 {
                alt_index |= 1 << bit;
            }
        }
        alt_indices.push(alt_index);
    }

    (main_index, alt_indices)
}

fn make_main_index(v: &[ProjectedValue]) -> u16 {
    let mut main_index = 0;

    for (i, pv) in v.iter().enumerate() {
        if pv.value >= 0.0 {
            main_index |= 1 << i;
        }
    }

    main_index
}

const VECTORS_COUNT: usize = 100_000;
const PERTURBATIONS_COUNT: usize = 100;
const QUERIES_COUNT: usize = 100;
const DIMENSION: usize = 640;
const TARGET_DIMENSION: usize = 16;
const PARTITIONS_COUNT: usize = 2usize.pow(TARGET_DIMENSION as u32);

fn serialize_results(results: Vec<(u16, usize, f32)>) -> String {
    let mut out = String::new();

    for (i, (partition_id, id, cs)) in results.into_iter().enumerate() {
        out.push_str(&format!(
            "#{:0>2}: {:0>5}:{:0>2} ({:.8}) [{:0>5} = {:016b}]\n",
            i + 1,
            id / 100,
            id % 100,
            cs,
            partition_id,
            partition_id,
        ));
    }

    out
}

// brute force
fn run_tests_bf(vectors: &[Vec<f32>], queries: &[Vec<f32>]) {
    println!("\nRunning brute force test");
    for (i, query) in queries.iter().enumerate().take(10) {
        println!("\nTest#{}", i + 1);
        let start = Instant::now();
        let mut top_matches: Vec<(usize, &[f32], f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, vec)| (i, vec.as_slice(), cosine_similarity(query, vec)))
            .collect();

        top_matches
            .sort_unstable_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        top_matches.truncate(20);
        println!("Finished in {:?}", start.elapsed());

        let results_serialized = serialize_results(
            top_matches
                .into_iter()
                .map(|(id, vec, cs)| {
                    let projected = project_vector_to_x(vec, TARGET_DIMENSION);
                    let main_index = make_main_index(&projected);
                    (main_index, id, cs)
                })
                .collect(),
        );
        fs::write(format!("result-bruteforce-{}", i), results_serialized).unwrap();
    }
}

#[derive(Debug, Default)]
struct Partition {
    vectors: Vec<usize>,
}

// project and partition
fn run_tests_pp(vectors: &[Vec<f32>], queries: &[Vec<f32>]) {
    println!("\nRunning project & partition test");
    let mut partitions: [Partition; PARTITIONS_COUNT] =
        std::array::from_fn(|_| Partition::default());

    println!("\nIndexing");

    let mut total_inserted_vectors = 0u64;

    let start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        let projected = project_vector_to_x(vector, TARGET_DIMENSION);
        let main_index = make_main_index(&projected);

        // let insert_in_main = !alt_index.contains(&main_index);

        // for index in alt_index {
        //     let partition = &mut partitions[index as usize];
        //     partition.vectors.push(i);
        //     total_inserted_vectors += 1;
        // }

        // if insert_in_main {
        let partition = &mut partitions[main_index as usize];
        partition.vectors.push(i);
        total_inserted_vectors += 1;
        // }
    }

    println!("Indexing finished in {:?}", start.elapsed());
    println!("Total inserted vectors: {}", total_inserted_vectors);
    println!(
        "Average vector replication rate: {}",
        total_inserted_vectors as f64 / (VECTORS_COUNT * PERTURBATIONS_COUNT) as f64
    );
    println!(
        "Average bucket size: {}",
        total_inserted_vectors as f64 / PARTITIONS_COUNT as f64
    );
    let mut total_query_time = Duration::ZERO;
    let mut total_alt_count = 0u64;

    for (i, query) in queries.iter().enumerate() {
        println!("\nTest#{}", i + 1);
        let start = Instant::now();
        let projected = project_vector_to_x(query, TARGET_DIMENSION);
        let (main_index, alt_index) = make_index(&projected);

        total_alt_count += alt_index.len() as u64;

        let mut top_matches: Vec<(u16, usize, f32)> = partitions[main_index as usize]
            .vectors
            .iter()
            .map(|&id| (main_index, id, cosine_similarity(query, &vectors[id])))
            .collect::<Vec<_>>();

        let query_alt_count = alt_index.len();

        println!("Alt count: {}", query_alt_count);

        let all_index_top_matches: Vec<(u16, usize, f32)> = alt_index
            .into_par_iter()
            .map(|index| {
                let partition = &partitions[index as usize];

                let mut top_matches = partition
                    .vectors
                    .iter()
                    .map(|&id| (index, id, cosine_similarity(query, &vectors[id])))
                    .collect::<Vec<_>>();

                top_matches.sort_unstable_by(|(_, _, a), (_, _, b)| {
                    b.partial_cmp(a).unwrap_or(Ordering::Equal)
                });
                top_matches.truncate(20);

                top_matches
            })
            .flatten()
            .collect();

        for (partition_id, id, cs) in all_index_top_matches {
            if top_matches.iter().any(|(_, id2, _)| id2 == &id) {
                continue;
            }
            top_matches.push((partition_id, id, cs));
        }

        top_matches
            .sort_unstable_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        top_matches.truncate(20);

        let elapsed = start.elapsed();

        total_query_time += elapsed;
        println!("Finished in {:?}", elapsed);

        let results_serialized = serialize_results(top_matches);
        fs::write(format!("result-partition-{}", i), results_serialized).unwrap();
    }

    let avg_query_time = total_query_time / queries.len() as u32;
    let avg_alt_count = total_alt_count as f32 / queries.len() as f32;

    println!("\nAverage query time: {:?}", avg_query_time);
    println!("Average alt count: {}", avg_alt_count);
    println!("Total execution time: {:?}", start.elapsed());
}

fn generate_random_vecs_and_save_to_file() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = thread_rng();

    println!("Generating vectors");

    let uniform = Uniform::new(-1.0, 1.0);
    let normal = Normal::new(0.0, 0.5).unwrap();

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(VECTORS_COUNT * PERTURBATIONS_COUNT);
    let mut base_vectors: Vec<Vec<f32>> = Vec::with_capacity(VECTORS_COUNT);

    for _ in 0..VECTORS_COUNT {
        let vec: Vec<f32> = (0..DIMENSION).map(|_| uniform.sample(&mut rng)).collect();

        for _ in 0..PERTURBATIONS_COUNT {
            let p_vec: Vec<f32> = vec
                .iter()
                .map(|&x| x * (1.0 + normal.sample(&mut rng)))
                .collect();
            vectors.push(p_vec);
        }

        base_vectors.push(vec);
    }

    println!("Generating queries");

    let queries: Vec<Vec<f32>> = (0..QUERIES_COUNT)
        .map(|_| {
            base_vectors
                .choose(&mut rng)
                .unwrap()
                .iter()
                .map(|&x| x * (1.0 + normal.sample(&mut rng)))
                .collect()
        })
        .collect();

    println!("Saving vectors");

    let vectors_serialized = bincode::serialize(&vectors).unwrap();
    fs::write("vectors.bin", vectors_serialized).unwrap();
    let queries_serialized = bincode::serialize(&queries).unwrap();
    fs::write("queries.bin", queries_serialized).unwrap();

    println!("Vectors saved");

    (vectors, queries)
}

fn load_vecs_from_file() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    println!("Loading vectors");
    let vectors_serialized = fs::read("vectors.bin").unwrap();
    let vectors = bincode::deserialize(&vectors_serialized).unwrap();
    let queries_serialized = fs::read("queries.bin").unwrap();
    let queries = bincode::deserialize(&queries_serialized).unwrap();
    println!("Vectors loaded");

    (vectors, queries)
}

fn main() {
    let (vectors, queries) =
        if Path::new("vectors.bin").exists() && Path::new("queries.bin").exists() {
            load_vecs_from_file()
        } else {
            generate_random_vecs_and_save_to_file()
        };

    run_tests_bf(&vectors, &queries);
    run_tests_pp(&vectors, &queries);
}
