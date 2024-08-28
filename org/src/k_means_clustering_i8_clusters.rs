// This is useful to determine the number the optimal quantization scheme for the collection, 
// based on sampling the initial vecs until the threshold value, we have set denominations ,
// either we decide the octal, quaternary or binary based on the number of dominant peaks in the clusters
// Note: this code uses i8 values.

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

fn generate_initial_centroids(x_vec: &[i8], k: usize) -> Vec<i8> {
    let mut rng = thread_rng();
    x_vec.choose_multiple(&mut rng, k).cloned().collect()
}

fn kmeans_scalar(
    x_vec: &[i8],
    initial_centroids: &[i8],
    iterations: usize,
) -> (Vec<i8>, Vec<usize>) {
    let k = initial_centroids.len();
    let mut centroids = initial_centroids.to_vec();
    let mut cluster_counts = vec![0; k];

    for _ in 0..iterations {
        let mut new_centroids = vec![0i32; k];
        let mut counts = vec![0u32; k];
        for &x in x_vec {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by_key(|(_, &c)| i8::abs_diff(x, c)) // Use i8::abs_diff instead
                .map(|(idx, _)| idx)
                .unwrap();
            new_centroids[nearest] += x as i32;
            counts[nearest] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centroids[i] = (new_centroids[i] / counts[i] as i32) as i8;
            }
        }
        cluster_counts = counts.iter().map(|&c| c as usize).collect();
    }
    (centroids, cluster_counts)
}
use std::arch::x86_64::*;
unsafe fn kmeans_simd(
    x_vec: &[i8],
    initial_centroids: &[i8],
    iterations: usize,
) -> (Vec<i8>, Vec<usize>) {
    let k = initial_centroids.len();
    let mut centroids = initial_centroids.to_vec();
    let mut cluster_counts = vec![0; k];

    for _ in 0..iterations {
        let mut new_centroids = vec![0i32; k];
        let mut counts = vec![0u32; k];
        for chunk in x_vec.chunks_exact(32) {
            let x_vec_avx = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let mut min_distances = _mm256_set1_epi8(i8::MAX);
            let mut min_indices = _mm256_setzero_si256();
            for (i, &centroid) in centroids.iter().enumerate() {
                let centroid_avx = _mm256_set1_epi8(centroid);
                let distances = _mm256_abs_epi8(_mm256_sub_epi8(x_vec_avx, centroid_avx));
                let mask = _mm256_cmpgt_epi8(min_distances, distances);
                min_distances = _mm256_blendv_epi8(min_distances, distances, mask);
                min_indices = _mm256_blendv_epi8(min_indices, _mm256_set1_epi8(i as i8), mask);
            }
            // Extract results
            let min_indices_array = std::mem::transmute::<__m256i, [u8; 32]>(min_indices);
            let chunk_array = std::mem::transmute::<__m256i, [i8; 32]>(x_vec_avx);
            for j in 0..32 {
                let index = min_indices_array[j] as usize;
                new_centroids[index] += chunk_array[j] as i32;
                counts[index] += 1;
            }
        }
        // Handle remaining elements
        for &x in x_vec.chunks_exact(32).remainder() {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by_key(|&(_, &c)| i8::abs_diff(x, c)) // Use i8::abs_diff here too
                .map(|(idx, _)| idx)
                .unwrap();
            new_centroids[nearest] += x as i32;
            counts[nearest] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centroids[i] = (new_centroids[i] / counts[i] as i32) as i8;
            }
        }
        cluster_counts = counts.iter().map(|&c| c as usize).collect();
    }
    (centroids, cluster_counts)
}

fn should_continue(counts: &[usize], threshold: f64, min_clusters: usize) -> bool {
    let mean_count = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    let threshold_count = mean_count * (1.0 + threshold);
    counts
        .iter()
        .filter(|&&count| count as f64 > threshold_count)
        .count()
        >= min_clusters
}

fn main() {
    let mut rng = rand::thread_rng();
    let mean = 0.0;
    let std_dev = 40.0;
    let normal = Normal::new(mean, std_dev).unwrap();

    let x_vec: Vec<i8> = (0..10000)
        .map(|_| {
            let value = normal.sample(&mut rng);
            if value < i8::MIN as f64 {
                i8::MIN
            } else if value > i8::MAX as f64 {
                i8::MAX
            } else {
                value as i8
            }
        })
        .collect();

    let iterations = 32;
    let threshold = 0.0; // 16% above mean

    // First iteration with k = 8
    let initial_centroids_8 = generate_initial_centroids(&x_vec, 8);
    let (centroids_8, counts_8) = kmeans_scalar(&x_vec, &initial_centroids_8, iterations);
    println!("K=8 centroids: {:?}", centroids_8);
    println!("K=8 cluster counts: {:?}", counts_8);

    if should_continue(&counts_8, threshold, 4) {
        // Second iteration with k = 4
        let initial_centroids_4 = generate_initial_centroids(&x_vec, 4);
        let (centroids_4, counts_4) = kmeans_scalar(&x_vec, &initial_centroids_4, iterations);
        println!("K=4 centroids: {:?}", centroids_4);
        println!("K=4 cluster counts: {:?}", counts_4);

        if should_continue(&counts_4, threshold, 2) {
            // Third iteration with k = 2
            let initial_centroids_2 = generate_initial_centroids(&x_vec, 2);
            let (centroids_2, counts_2) = kmeans_scalar(&x_vec, &initial_centroids_2, iterations);
            println!("K=2 centroids: {:?}", centroids_2);
            println!("K=2 cluster counts: {:?}", counts_2);
        }
    }

    // SIMD version
    unsafe {
        // First iteration with k = 8
        let (simd_centroids_8, simd_counts_8) =
            kmeans_simd(&x_vec, &initial_centroids_8, iterations);
        println!("\nSIMD K=8 centroids: {:?}", simd_centroids_8);
        println!("SIMD K=8 cluster counts: {:?}", simd_counts_8);

        if should_continue(&simd_counts_8, threshold, 4) {
            // Second iteration with k = 4
            let initial_centroids_4 = generate_initial_centroids(&x_vec, 4);
            let (simd_centroids_4, simd_counts_4) =
                kmeans_simd(&x_vec, &initial_centroids_4, iterations);
            println!("SIMD K=4 centroids: {:?}", simd_centroids_4);
            println!("SIMD K=4 cluster counts: {:?}", simd_counts_4);

            if should_continue(&simd_counts_4, threshold, 2) {
                // Third iteration with k = 2
                let initial_centroids_2 = generate_initial_centroids(&x_vec, 2);
                let (simd_centroids_2, simd_counts_2) =
                    kmeans_simd(&x_vec, &initial_centroids_2, iterations);
                println!("SIMD K=2 centroids: {:?}", simd_centroids_2);
                println!("SIMD K=2 cluster counts: {:?}", simd_counts_2);
            }
        }
    }
}
