use std::arch::x86_64::*;

// Update the main clustering function
unsafe fn simd_kmeans_clustering(x_vec: &[Vec<u8>], k: usize) -> Vec<u8> {
    assert!(k <= 256, "K should be less than or equal to 256");
    let iterations = 32;
    // Initialize centroids
    let mut centroids = initialize_centroids(k);
    for _ in 0..iterations {
        // Convert centroids to binary and create SIMD vectors
        let centroid_vectors = create_centroid_vectors(&centroids);
        // Process data in SIMD chunks and update centroids
        process_data_simd(x_vec, &centroid_vectors, &mut centroids);
    }
    centroids
}

fn initialize_centroids(k: usize) -> Vec<u8> {
    (0..k).map(|i| (i * 255 / (k - 1)) as u8).collect()
}

#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn create_centroid_vectors(centroids: &[u8]) -> Vec<Vec<__m256i>> {
    let mut all_centroid_vectors = Vec::with_capacity(centroids.len());

    for &centroid in centroids {
        let mut centroid_vectors = Vec::with_capacity(8);
        for bit in 0..8 {
            let mask = if (centroid & (1 << bit)) != 0 {
                unsafe { _mm256_set1_epi32(-1) } // All 1s
            } else {
                unsafe { _mm256_setzero_si256() } // All 0s
            };
            centroid_vectors.push(mask);
        }
        all_centroid_vectors.push(centroid_vectors);
    }

    all_centroid_vectors
}
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn process_data_simd(
    x: &[Vec<u8>],
    centroid_vectors: &[Vec<__m256i>],
    centroids: &mut Vec<u8>,
) {
    assert_eq!(x.len(), 8, "Expected 8 bit planes");

    let num_centroids = centroids.len();
    // Initialize counts for each bit of each centroid
    let mut centroid_counts = vec![vec![0u64; 8]; num_centroids];

    for bit in (0..8).rev() {
        // Start from MSB (7) to LSB (0)
        let mut count = 0u64;
        for i in (0..x[bit].len()).step_by(32) {
            // Load 32 bytes (256 bits) of data from x vector
            let x_bits = _mm256_loadu_si256(x[bit][i..].as_ptr() as *const __m256i);

            //  AND and count ones
            for (centroid_index, centroid_vector) in centroid_vectors.iter().enumerate() {
                let x_and = _mm256_and_si256(x_bits, centroid_vector[bit]);
                count = count_ones_simd_avx2_256i(x_and);
                centroid_counts[centroid_index][bit] += count;
            }
        }
    }

    // Adjust centroids based on counts
    // ***** Todo *****
    // depending on if the count is above/below 128 ,
    // then increment/decrement the centroid (not sure which is which)
    // Its 128 for each step of 32 (in this example there are 64, so 256 is the mid point)
    for (i, centroid_count) in centroid_counts.iter().enumerate() {

        let sum : u64 = centroid_count.iter().sum();

        let average = sum / 8;

        if average >= 128 {
            centroids[i] = cmp::min(254, centroids[i]) + 1;
        } else {
            centroids[i] = cmp::max(1, centroids[i]) - 1;
        }
    }
}

#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn count_ones_simd_avx2_256i(input: __m256i) -> u64 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let lo = _mm256_and_si256(input, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(input, 4), low_mask);
    let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
    let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
    let sum = _mm256_add_epi8(popcnt_lo, popcnt_hi);
    // Sum all bytes using horizontal addition
    let sum_16 = _mm256_sad_epu8(sum, _mm256_setzero_si256());
    let sum_64 = _mm256_add_epi64(
        _mm256_unpacklo_epi64(sum_16, _mm256_setzero_si256()),
        _mm256_unpackhi_epi64(sum_16, _mm256_setzero_si256()),
    );
    // Extract and add the final sum
    _mm256_extract_epi64(sum_64, 0) as u64 + _mm256_extract_epi64(sum_64, 2) as u64
}

use rand::Rng;
use std::arch::x86_64::*;

fn main() {
    let num_vectors = 8; // Number of vectors in x_vec
    let vector_length = 64; // Length of each vector, must be multiple of 32
    let k = 4;

    // Generate random x_vec and y_vec
    let mut rng = rand::thread_rng();
    let x_vec = generate_random_vectors(&mut rng, num_vectors, vector_length);

    // Print a sample of the generated data
    println!(" x_vec: {:?}", &x_vec[0][0..]);

    // Check if AVX2 is supported
    if is_x86_feature_detected!("avx2") {
        let result = unsafe { simd_kmeans_clustering(&x_vec, k) };
        println!("Resulting centroids: {:?}", result);
    } else {
        println!("AVX2 is not supported on this CPU");
    }
}

fn generate_random_vectors(
    rng: &mut impl Rng,
    num_vectors: usize,
    vector_length: usize,
) -> Vec<Vec<u8>> {
    (0..num_vectors)
        .map(|_| (0..vector_length).map(|_| rng.gen::<u8>()).collect())
        .collect()
}
