#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rand::{seq::SliceRandom, thread_rng};

#[allow(dead_code)]
pub fn should_continue(counts: &[usize], threshold: f64, min_clusters: usize) -> bool {
    let mean_count = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    let threshold_count = mean_count * (1.0 + threshold);
    counts
        .iter()
        .filter(|&&count| count as f64 > threshold_count)
        .count()
        >= min_clusters
}

#[allow(dead_code)]
pub fn concat_vectors(vectors: &[&[f32]]) -> Vec<i8> {
    let mut vec = Vec::new();
    for &vector in vectors {
        for value in vector {
            vec.push((value * 127.0) as i8)
        }
    }
    vec
}

#[allow(dead_code)]
pub fn generate_initial_centroids(x_vec: &[i8], k: usize) -> Vec<i8> {
    let mut rng = thread_rng();
    x_vec.choose_multiple(&mut rng, k).cloned().collect()
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
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

#[allow(dead_code)]
pub fn kmeans(x_vec: &[i8], initial_centroids: &[i8], iterations: usize) -> (Vec<i8>, Vec<usize>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            return unsafe { kmeans_simd(x_vec, initial_centroids, iterations) };
        }
    }

    kmeans_scalar(x_vec, initial_centroids, iterations)
}
