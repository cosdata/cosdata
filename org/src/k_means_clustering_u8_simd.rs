use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

// Scalar implementation
fn kmeans_scalar(x_vec: &[u8], k: usize, iterations: usize) -> (Vec<u8>, Vec<usize>) {
    let mut rng = thread_rng();
    let mut centroids: Vec<u8> = x_vec.choose_multiple(&mut rng, k).cloned().collect();
    let mut cluster_counts = vec![0; k];
    
    for _ in 0..iterations {
        let mut new_centroids = vec![0u32; k];
        let mut counts = vec![0u32; k];
        for &x in x_vec {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by_key(|(_, &c)| x.abs_diff(c) as u16)
                .map(|(idx, _)| idx)
                .unwrap();
            new_centroids[nearest] += x as u32;
            counts[nearest] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centroids[i] = (new_centroids[i] / counts[i]) as u8;
            }
        }
        cluster_counts = counts.iter().map(|&c| c as usize).collect();
    }
    (centroids, cluster_counts)
}

unsafe fn kmeans_simd(x_vec: &[u8], k: usize, iterations: usize) -> (Vec<u8>, Vec<usize>) {
    use std::arch::x86_64::*;
    let mut rng = thread_rng();
    let mut centroids: Vec<u8> = x_vec.choose_multiple(&mut rng, k).cloned().collect();
    let mut cluster_counts = vec![0; k];
    
    for _ in 0..iterations {
        let mut new_centroids = vec![0u32; k];
        let mut counts = vec![0u32; k];
        for chunk in x_vec.chunks_exact(32) {
            let x_vec_avx = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let mut min_distances = _mm256_set1_epi8(i8::MAX);
            let mut min_indices = _mm256_setzero_si256();
            for (i, &centroid) in centroids.iter().enumerate() {
                let centroid_avx = _mm256_set1_epi8(centroid as i8);
                let distances = _mm256_abs_epi8(_mm256_sub_epi8(x_vec_avx, centroid_avx));
                let mask = _mm256_cmpgt_epi8(min_distances, distances);
                min_distances = _mm256_blendv_epi8(min_distances, distances, mask);
                min_indices = _mm256_blendv_epi8(min_indices, _mm256_set1_epi8(i as i8), mask);
            }
            // Extract results
            let min_indices_array = std::mem::transmute::<__m256i, [u8; 32]>(min_indices);
            let chunk_array = std::mem::transmute::<__m256i, [u8; 32]>(x_vec_avx);
            for j in 0..32 {
                let index = min_indices_array[j] as usize;
                new_centroids[index] += chunk_array[j] as u32;
                counts[index] += 1;
            }
        }
        // Handle remaining elements
        for &x in x_vec.chunks_exact(32).remainder() {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by_key(|&(_, &c)| x.abs_diff(c) as u16)
                .map(|(idx, _)| idx)
                .unwrap();
            new_centroids[nearest] += x as u32;
            counts[nearest] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centroids[i] = (new_centroids[i] / counts[i]) as u8;
            }
        }
        cluster_counts = counts.iter().map(|&c| c as usize).collect();
    }
    (centroids, cluster_counts)
}

fn main() {
    let mut rng = rand::thread_rng();
    // Parameters for the normal distribution
    let mean = 128.0;  // Center of the distribution (middle of u8 range)
    let std_dev = 40.0;  // Standard deviation
    // Create a normal distribution
    let normal = Normal::new(mean, std_dev).unwrap();
    // Generate a vector of 10000 random u8 values following a normal distribution
    let x_vec: Vec<u8> = (0..10000)
        .map(|_| {
            let value = normal.sample(&mut rng);
            value as u8  // Ensure values are within u8 range
        })
        .collect();
    let k = 4;
    let iterations = 32;
    
    // Run scalar version
    let (scalar_result, scalar_counts) = kmeans_scalar(&x_vec, k, iterations);
    println!("Scalar K-means centroids: {:?}", scalar_result);
    println!("Scalar K-means cluster counts: {:?}", scalar_counts);
    
    unsafe {
        let (simd_result, simd_counts) = kmeans_simd(&x_vec, k, iterations);
        println!("SIMD K-means centroids: {:?}", simd_result);
        println!("SIMD K-means cluster counts: {:?}", simd_counts);
    }
}
