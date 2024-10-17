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
// 01. 4541:74 (0.8526581)
// 02. 4541:98 (0.8510747)
// 03. 4541:13 (0.8489201)
// 04. 4541:85 (0.84865904)
// 05. 4541:6  (0.8478159)
// 06. 4541:95 (0.84775466)
// 07. 4541:44 (0.8470717)
// 08. 4541:73 (0.8470653)
// 09. 4541:3  (0.8462134)
// 10. 4541:29 (0.84520525)
// 11. 4541:49 (0.8412987)
// 12. 4541:60 (0.83847)
// 13. 4541:76 (0.83761704)
// 14. 4541:81 (0.8369249)
// 15. 4541:93 (0.83678424)
// 16. 4541:89 (0.83608675)
// 17. 4541:7  (0.8360864)
// 18. 4541:96 (0.8346243)
// 19. 4541:68 (0.8341953)
// 20. 4541:59 (0.83310014)
//
// ## Project & Partition
//
// 01. 4541:74 (0.8526581)
// 02. 4541:98 (0.8510747)
// 03. 4541:13 (0.8489201)
// 04. 4541:6  (0.8478159)
// 05. 4541:95 (0.84775466)
// 06. 4541:44 (0.8470717)
// 07. 4541:73 (0.8470653)
// 08. 4541:29 (0.84520525)
// 09. 4541:49 (0.8412987)
// 10. 4541:60 (0.83847)
// 11. 4541:76 (0.83761704)
// 12. 4541:81 (0.8369249)
// 13. 4541:93 (0.83678424)
// 14. 4541:89 (0.83608675)
// 15. 4541:96 (0.8346243)
// 16. 4541:68 (0.8341953)
// 17. 4541:59 (0.83310014)
// 18. 4541:48 (0.82951933)
// 19. 4541:51 (0.82864046)
// 20. 4541:82 (0.8280492)
//
// ## Performance
//
// Brute force:
//   - Query time: ~ 2.47s
//
// Project & Partition:
//   - Index creation time: ~ 33s
//   - Query time: ~ 3ms

use std::{cmp::Ordering, fs, path::Path, time::Instant};

use rand::prelude::*;
use rand_distr::{Normal, Uniform};

#[derive(Clone)]
struct ProjectedValue {
    value: f32,
    is_sensitive: bool,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

fn project_to_3d(x: f32, y: f32) -> f32 {
    let theta = y.atan2(x);
    theta / std::f32::consts::PI
}

fn is_sensitive_pair(x: f32, y: f32) -> bool {
    x.abs() < 0.2 && y.abs() > 0.8
}

fn project_vector_to_x(v: &[f32], x: usize) -> Vec<ProjectedValue> {
    let mut projected = v
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                ProjectedValue {
                    value: project_to_3d(chunk[0], chunk[1]),
                    is_sensitive: is_sensitive_pair(chunk[0], chunk[1]),
                }
            } else {
                ProjectedValue {
                    value: chunk[0],
                    is_sensitive: false,
                }
            }
        })
        .collect::<Vec<_>>();

    while projected.len() > x {
        if projected.len() / 2 < x {
            let pairs_to_project = projected.len() - x;

            let mut new_projected = Vec::with_capacity(x);

            for j in 0..pairs_to_project {
                let first = &projected[2 * j];
                let second = &projected[2 * j + 1];
                let z = project_to_3d(first.value, second.value);
                let is_sensitive = first.is_sensitive
                    || second.is_sensitive
                    || is_sensitive_pair(first.value, second.value);
                new_projected.push(ProjectedValue {
                    value: z,
                    is_sensitive,
                });
            }

            new_projected.extend(projected[2 * pairs_to_project..].iter().cloned());

            projected = new_projected;
        } else {
            let mut new_projected = Vec::with_capacity(projected.len() / 2);

            for chunk in projected.chunks(2) {
                if chunk.len() == 2 {
                    let z = project_to_3d(chunk[0].value, chunk[1].value);
                    let is_sensitive = chunk[0].is_sensitive
                        || chunk[1].is_sensitive
                        || is_sensitive_pair(chunk[0].value, chunk[1].value);
                    new_projected.push(ProjectedValue {
                        value: z,
                        is_sensitive,
                    });
                } else {
                    new_projected.push(chunk[0].clone());
                }
            }

            if new_projected.len() < x {
                new_projected.extend(projected[new_projected.len()..].iter().cloned());
            }

            projected = new_projected;
        }
    }

    projected
}

fn make_index(v: &[ProjectedValue]) -> (u16, Vec<u16>) {
    let mut main_index = 0;
    let mut main_mask = 0;
    let mut sway_bits = Vec::new();

    for (i, pv) in v.iter().enumerate() {
        if pv.value >= 0.0 {
            main_index |= 1 << i;
        }
        if pv.value >= 0.1 {
            main_mask |= 1 << i;
        } else if pv.is_sensitive {
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
const DIMENSION: usize = 320;
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

#[derive(Debug, Default)]
struct Partition {
    vectors: Vec<usize>,
}

// project and partition
fn run_tests_pp(vectors: &[Vec<f32>], queries: &[Vec<f32>]) {
    println!("\nRunning project & partition test");
    let mut partitions: [Partition; 2usize.pow(TARGET_DIMENSION as u32) + 1] =
        std::array::from_fn(|_| Partition::default());

    println!("\nIndexing");

    let start = Instant::now();
    for (i, vector) in vectors.into_iter().enumerate() {
        let projected = project_vector_to_x(&vector, TARGET_DIMENSION);
        let (main_index, alt_index) = make_index(&projected);

        let insert_in_main = !alt_index.contains(&main_index);

        for index in &alt_index {
            let partition = &mut partitions[*index as usize];
            partition.vectors.push(i);
        }

        if insert_in_main {
            let partition = &mut partitions[main_index as usize];
            partition.vectors.push(i);
        }
    }
    println!("Indexing finished in {:?}", start.elapsed());

    for (i, query) in queries.into_iter().enumerate() {
        println!("\nTest#{}", i + 1);
        let start = Instant::now();
        let projected = project_vector_to_x(&query, TARGET_DIMENSION);
        let (main_index, _alt_index) = make_index(&projected);

        let mut top_matches: Vec<(usize, f32)> = partitions[main_index as usize]
            .vectors
            .iter()
            .map(|&id| (id, cosine_similarity(query, &vectors[id])))
            .collect::<Vec<_>>();

        // for index in alt_index {
        //     let partition = &partitions[index as usize];
        //     for id in &partition.vectors {
        //         if top_matches.iter().position(|(id2, _)| id2 == id).is_some() {
        //             continue;
        //         }
        //         let cs = cosine_similarity(query, &vectors[*id]);
        //         top_matches.push((*id, cs));
        //     }
        // }

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
        fs::write(format!("result-partition-{}", i), results_serialized).unwrap();
    }

    println!("\nTotal execution time: {:?}", start.elapsed());
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
