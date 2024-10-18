// # Project & Partition comparison with brute force.
//
// Generates 50k base vectors and 100 perturbations for each vector (50k x 100 = 5M).
// Randomly picks 100 vectors from the base vectors and generates perturbation of each,
// and saves them as query vectors.
//
// A sample run with same `vectors` and `query`
//
// ## Brute force
//
// 01. 43148:13 (0.8465757)
// 02. 43148:83 (0.8402604)
// 03. 43148:52 (0.8391722)
// 04. 43148:41 (0.83780986)
// 05. 43148:95 (0.8375291)
// 06. 43148:49 (0.8367006)
// 07. 43148:6  (0.836649)
// 08. 43148:19 (0.8357567)
// 09. 43148:53 (0.83417034)
// 10. 43148:16 (0.83392346)
// 11. 43148:17 (0.83354986)
// 12. 43148:29 (0.83285147)
// 13. 43148:9  (0.832588)
// 14. 43148:48 (0.8313629)
// 15. 43148:25 (0.8308406)
// 16. 43148:50 (0.8307434)
// 17. 43148:18 (0.83015645)
// 18. 43148:82 (0.82967573)
// 19. 43148:5  (0.8270634)
// 20. 43148:24 (0.8268056)
//
// ## Project & Partition
//
// 01. 43148:13 (0.8465757)
// 02. 43148:83 (0.8402604)
// 03. 43148:52 (0.8391722)
// 04. 43148:41 (0.83780986)
// 05. 43148:95 (0.8375291)
// 06. 43148:49 (0.8367006)
// 07. 43148:6  (0.836649)
// 08. 43148:19 (0.8357567)
// 09. 43148:16 (0.83392346)
// 10. 43148:17 (0.83354986)
// 11. 43148:29 (0.83285147)
// 12. 43148:9  (0.832588)
// 13. 43148:48 (0.8313629)
// 14. 43148:25 (0.8308406)
// 15. 43148:50 (0.8307434)
// 16. 43148:82 (0.82967573)
// 17. 43148:5  (0.8270634)
// 18. 43148:24 (0.8268056)
// 19. 43148:35 (0.82625026)
// 20. 43148:88 (0.82524896)
//
// ## Performance
//
// Brute force:
//   - Query time: ~ 2.47s
//
// Project & Partition:
//   - Index creation time: ~ 46s
//   - Query time: ~ 342ms

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
        y.abs() > 0.9
    } else {
        (y / x).abs() > 0.9
    }
}

fn calculate_weight(iteration: u8) -> f32 {
    (iteration as f32).powi(2) / 1.618034 // Golden ratio approximation
}

fn project_to_3d(x: f32, y: f32) -> f32 {
    let theta = y.atan2(x);
    theta / std::f32::consts::PI
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
    let max_possible_weight: f32 = (1..=max_iterations).map(|i| calculate_weight(i)).sum();
    let threshold = max_possible_weight / 1.25;

    for (i, pv) in v.iter().enumerate() {
        if pv.value >= 0.0 {
            main_index |= 1 << i;
        }
        if pv.value >= 0.1 {
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

const VECTORS_COUNT: usize = 50_000;
const PERTURBATIONS_COUNT: usize = 100;
const QUERIES_COUNT: usize = 100;
const DIMENSION: usize = 640;
const TARGET_DIMENSION: usize = 16;

fn serialize_results(results: Vec<(usize, f32)>) -> String {
    let mut out = String::new();

    for (i, (id, cs)) in results.into_iter().enumerate() {
        out.push_str(&format!("#{}: {}:{} ({})\n", i + 1, id / 100, id % 100, cs));
    }

    out
}

// brute force
fn run_tests_bf(vectors: &[Vec<f32>], queries: &[Vec<f32>]) {
    println!("\nRunning brute force test");
    for (i, query) in queries.into_iter().enumerate() {
        println!("\nTest#{}", i + 1);
        let start = Instant::now();
        let mut top_matches: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, vec)| (i, cosine_similarity(&query, vec)))
            .collect();

        top_matches.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        top_matches.truncate(20);
        println!("Finished in {:?}", start.elapsed());

        let mut alt_count = 0;

        for (id, _) in top_matches.iter() {
            let vec = &vectors[*id];
            let projected = project_vector_to_x(vec, TARGET_DIMENSION);
            let (_, alt_index) = make_index(&projected);

            alt_count += alt_index.len();
        }

        println!("Avg alt count: {}", alt_count as f32 / 20.0);

        let results_serialized = serialize_results(top_matches);
        fs::write(format!("result-bruteforce-{}", i), results_serialized).unwrap();
    }
}

const PARTITIONS_COUNT: usize = 2usize.pow(TARGET_DIMENSION as u32) + 1;

#[derive(Debug)]
struct Partition {
    vectors: Vec<usize>,
}

impl Default for Partition {
    fn default() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }
}

// project and partition
fn run_tests_pp(vectors: &[Vec<f32>], queries: &[Vec<f32>]) {
    println!("\nRunning project & partition test");
    let mut partitions: [Partition; PARTITIONS_COUNT] =
        std::array::from_fn(|_| Partition::default());

    println!("\nIndexing");

    let mut total_inserted_vectors = 0u64;

    let start = Instant::now();
    for (i, vector) in vectors.into_iter().enumerate() {
        let projected = project_vector_to_x(&vector, TARGET_DIMENSION);
        let (main_index, alt_index) = make_index(&projected);

        let insert_in_main = !alt_index.contains(&main_index);

        for index in &alt_index {
            let partition = &mut partitions[*index as usize];
            partition.vectors.push(i);
            total_inserted_vectors += 1;
        }

        if insert_in_main {
            let partition = &mut partitions[main_index as usize];
            partition.vectors.push(i);
            total_inserted_vectors += 1;
        }
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

    for (i, query) in queries.into_iter().enumerate() {
        println!("\nTest#{}", i + 1);
        let start = Instant::now();
        let projected = project_vector_to_x(&query, TARGET_DIMENSION);
        let (main_index, alt_index) = make_index(&projected);
        let mut skip_partitions = vec![false; PARTITIONS_COUNT];

        skip_partitions[main_index as usize] = true;

        let mut top_matches: Vec<(usize, f32)> = partitions[main_index as usize]
            .vectors
            .iter()
            .map(|&id| (id, cosine_similarity(query, &vectors[id])))
            .collect::<Vec<_>>();

        let query_alt_count = alt_index.len();

        println!("Alt count: {}", query_alt_count);

        let all_index_top_matches: Vec<(usize, f32)> = alt_index
            .into_par_iter()
            .map(|index| {
                let partition = &partitions[index as usize];

                let mut top_matches = partition
                    .vectors
                    .iter()
                    .map(|&id| (id, cosine_similarity(query, &vectors[id])))
                    .collect::<Vec<_>>();

                top_matches
                    .sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
                top_matches.truncate(20);

                top_matches
            })
            .flatten()
            .collect();

        for (id, cs) in all_index_top_matches {
            if top_matches.iter().position(|(id2, _)| id2 == &id).is_some() {
                continue;
            }
            top_matches.push((id, cs));
        }

        top_matches.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        top_matches.truncate(20);

        let elapsed = start.elapsed();

        total_query_time += elapsed;
        println!("Finished in {:?}", elapsed);

        let results_serialized = serialize_results(top_matches);
        fs::write(format!("result-partition-{}", i), results_serialized).unwrap();
    }

    let avg_query_time = total_query_time / queries.len() as u32;

    println!("\nAverage query time: {:?}", avg_query_time);
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
