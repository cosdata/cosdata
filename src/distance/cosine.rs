use serde::{Deserialize, Serialize};

use super::{DistanceError, DistanceFunction};
use crate::storage::Storage;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub struct CosineDistance(pub f32);

impl DistanceFunction for CosineDistance {
    type Item = Self;
    fn calculate(&self, _x: &Storage, _y: &Storage) -> Result<Self::Item, DistanceError> {
        // placeholder method to be implemented
        Err(DistanceError::CalculationError)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub struct CosineSimilarity(pub f32);

impl DistanceFunction for CosineSimilarity {
    type Item = Self;
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<Self::Item, DistanceError> {
        match (x, y) {
            (
                Storage::UnsignedByte {
                    mag: _mag_x,
                    quant_vec: _vec_x,
                },
                Storage::UnsignedByte {
                    mag: _mag_y,
                    quant_vec: _vec_y,
                },
            ) => {
                // Implement cosine similarity for UnsignedByte storage
                //unimplemented!("Cosine similarity for UnsignedByte not implemented yet")
                Ok(CosineSimilarity(0.0))
            }
            (
                Storage::SubByte {
                    mag: x_mag,
                    quant_vec: x_vec,
                    resolution: x_res,
                },
                Storage::SubByte {
                    mag: y_mag,
                    quant_vec: y_vec,
                    resolution: y_res,
                },
            ) => {
                if x_res != y_res {
                    return Err(DistanceError::StorageMismatch);
                }
                match *x_res {
                    1 => {
                        let dot_product = dot_product_binary(x_vec, y_vec, *x_res)
                            .expect("Failed computing dot product");
                        cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag)
                    }
                    2 => {
                        let dot_product = dot_product_quaternary(x_vec, y_vec, *x_res);
                        cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag)
                    }
                    _ => Err(DistanceError::CalculationError),
                }
            }
            (Storage::HalfPrecisionFP { .. }, Storage::HalfPrecisionFP { .. }) => {
                // Implement cosine similarity for HalfPrecisionFP storage
                unimplemented!("Cosine similarity for HalfPrecisionFP not implemented yet")
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}

fn dot_product_binary(
    x_vec: &[Vec<u8>],
    y_vec: &[Vec<u8>],
    resolution: u8,
) -> Result<f32, DistanceError> {
    let parts = 2_usize.pow(resolution as u32);

    let mut final_result: usize = 0;

    for index in 0..parts {
        let sum: usize = x_vec[index]
            .iter()
            .zip(&y_vec[index])
            .map(|(&x_item, &y_item)| (x_item & y_item).count_ones() as usize)
            .sum();
        final_result += sum << index;
    }

    let dot_product = final_result;
    Ok(dot_product as f32)
}

fn cosine_similarity_from_dot_product(
    dot_product: f32,
    mag_x: u32,
    mag_y: u32,
) -> Result<CosineSimilarity, DistanceError> {
    let denominator = (mag_x as f32).sqrt() * (mag_y as f32).sqrt();
    if denominator == 0.0 {
        Err(DistanceError::CalculationError)
    } else {
        Ok(CosineSimilarity(dot_product / denominator))
    }
}

fn dot_product_quaternary(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], resolution: u8) -> f32 {
    assert_eq!(resolution, 2);

    let dot_product: u32 = x_vec[0]
        .iter()
        .zip(&x_vec[1])
        .zip(y_vec[0].iter().zip(&y_vec[1]))
        .enumerate()
        .map(|(_i, ((&x_lsb, &x_msb), (&y_lsb, &y_msb)))| {
            let lsbs = (x_lsb & y_lsb).count_ones();
            let mid1 = x_lsb & y_msb;
            let mid2 = y_lsb & x_msb;
            let carry = (mid1 & mid2).count_ones();
            let msbs = (x_msb & y_msb).count_ones();
            let mid = (mid1 ^ mid2).count_ones();

            let result = (msbs << 2) + (carry << 2) + (mid << 1) + lsbs;
            result
        })
        .sum();

    dot_product as f32
}
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn dot_product_quaternary_avx2(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], resolution: u8) -> f32 {
    assert_eq!(resolution, 2);
    assert!(
        x_vec[0].len() % 32 == 0,
        "Vector length must be a multiple of 32"
    );

    let mut dot_product = 0u64;

    for i in (0..x_vec[0].len()).step_by(32) {
        let x_lsb = _mm256_loadu_si256(x_vec[0][i..].as_ptr() as *const __m256i);
        let x_msb = _mm256_loadu_si256(x_vec[1][i..].as_ptr() as *const __m256i);
        let y_lsb = _mm256_loadu_si256(y_vec[0][i..].as_ptr() as *const __m256i);
        let y_msb = _mm256_loadu_si256(y_vec[1][i..].as_ptr() as *const __m256i);

        let lsbs = _mm256_and_si256(x_lsb, y_lsb);
        let mid1 = _mm256_and_si256(x_lsb, y_msb);
        let mid2 = _mm256_and_si256(y_lsb, x_msb);
        let msbs = _mm256_and_si256(x_msb, y_msb);

        let carry = _mm256_and_si256(mid1, mid2);
        let mid = _mm256_xor_si256(mid1, mid2);

        let lsbs_count = count_ones_simd_avx2_256i(lsbs);
        let carry_count = count_ones_simd_avx2_256i(carry);
        let msbs_count = count_ones_simd_avx2_256i(msbs);
        let mid_count = count_ones_simd_avx2_256i(mid);

        // Perform shifts and additions using u64 arithmetic
        let msbs_shifted = msbs_count << 2;
        let carry_shifted = carry_count << 2;
        let mid_shifted = mid_count << 1;

        dot_product += msbs_shifted + carry_shifted + mid_shifted + lsbs_count;
    }
    dot_product as f32
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

#[target_feature(enable = "avx2")]
unsafe fn quaternary_weighted_simd_avx2(data: *const u8, n: usize, lookup: &[u8; 32]) -> u64 {
    let mut i = 0;
    let lookup_vec = _mm256_loadu_si256(lookup.as_ptr() as *const __m256i);
    let low_mask = _mm256_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_si256();

    while i + 32 < n {
        let mut local = _mm256_setzero_si256();
        for _ in 0..255 / 8 {
            if i + 32 >= n {
                break;
            }
            let vec = _mm256_loadu_si256(data.add(i) as *const __m256i);
            let lo = _mm256_and_si256(vec, low_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
            let popcnt1 = _mm256_shuffle_epi8(lookup_vec, lo);
            let popcnt2 = _mm256_shuffle_epi8(lookup_vec, hi);
            local = _mm256_add_epi8(local, popcnt1);
            local = _mm256_add_epi8(local, popcnt2);
            i += 32;
        }
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    let mut result = 0u64;
    result += _mm256_extract_epi64(acc, 0) as u64;
    result += _mm256_extract_epi64(acc, 1) as u64;
    result += _mm256_extract_epi64(acc, 2) as u64;
    result += _mm256_extract_epi64(acc, 3) as u64;

    // Handle remaining bytes using lookup
    for j in i..n {
        let byte = *data.add(j);
        result += lookup[(byte & 0x0f) as usize] as u64;
        result += lookup[(byte >> 4) as usize] as u64;
    }

    result
}

// Outer function to create lookup table and call the SIMD function
pub fn quaternary_weighted_wrapper(data: &[u8]) -> u64 {
    let lookup: [u8; 32] = [
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // repeated
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    ];
    unsafe { quaternary_weighted_simd_avx2(data.as_ptr(), data.len(), &lookup) }
}

// pub fn octal_weighted_wrapper_old(data: &[u8]) -> u64 {
//     // Initialize lookup tables
//     let mut lookup0 = [0u8; 32];
//     let mut lookup1 = [0u8; 32];

//     for i in 0..64 {
//         let popcount = (i as u8 & 0x3F).count_ones() as u8; // Only count lower 6 bits
//         if i < 32 {
//             lookup0[i] = popcount;
//         } else {
//             lookup1[i - 32] = popcount;
//         }
//     }

//     unsafe { octal_weighted_simd_avx2(data.as_ptr(), data.len(), &lookup0, &lookup1) }
// }
pub fn octal_weighted_wrapper(data: &[u8]) -> u64 {
    // Initialize lookup table
    let mut lookup = [0u8; 64];
    for i in 0..64 {
        lookup[i] = i.count_ones() as u8;
    }

    unsafe { octal_weighted_simd_avx2(data.as_ptr(), data.len(), &lookup) }
}
unsafe fn octal_weighted_simd_avx2(data: *const u8, n: usize, lookup: &[u8; 64]) -> u64 {
    let mut i = 0;
    // Load 16 bytes and duplicate them in a 256-bit register
    let lookup_vec0 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(lookup.as_ptr() as *const __m128i));
    let lookup_vec1 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(lookup.as_ptr().add(16) as *const __m128i));
    let lookup_vec2 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(lookup.as_ptr().add(32) as *const __m128i));
    let lookup_vec3 =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(lookup.as_ptr().add(48) as *const __m128i));

    let low_mask = _mm256_set1_epi8(0x0f); // 4 bits mask
    let mut acc = _mm256_setzero_si256();
    while i + 32 < n {
        let mut local = _mm256_setzero_si256();
        for _ in 0..255 / 8 {
            if i + 32 >= n {
                break;
            }
            let vec = _mm256_loadu_si256(data.add(i) as *const __m256i);
            let vec_masked = _mm256_and_si256(vec, _mm256_set1_epi8(0x3F)); // Mask to lower 6 bits

            let lo = _mm256_and_si256(vec_masked, low_mask);
            let hi = _mm256_srli_epi16(vec_masked, 4);

            let result0 = _mm256_shuffle_epi8(lookup_vec0, lo);
            let result1 = _mm256_shuffle_epi8(lookup_vec1, lo);
            let result2 = _mm256_shuffle_epi8(lookup_vec2, lo);
            let result3 = _mm256_shuffle_epi8(lookup_vec3, lo);

            let blend01 = _mm256_blendv_epi8(result0, result1, _mm256_slli_epi16(hi, 7));
            let blend23 = _mm256_blendv_epi8(result2, result3, _mm256_slli_epi16(hi, 7));
            let popcnt = _mm256_blendv_epi8(blend01, blend23, _mm256_slli_epi16(hi, 6));

            local = _mm256_add_epi8(local, popcnt);
            i += 32;
        }
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    let mut result = 0u64;
    result += _mm256_extract_epi64(acc, 0) as u64;
    result += _mm256_extract_epi64(acc, 1) as u64;
    result += _mm256_extract_epi64(acc, 2) as u64;
    result += _mm256_extract_epi64(acc, 3) as u64;

    // Handle remaining bytes
    while i < n {
        let byte = *data.add(i) & 0x3F; // Mask to lower 6 bits
        result += lookup[byte as usize] as u64;
        i += 1;
    }

    result
}

// Scalar implementation for comparison

fn scalar_u6_count_ones(data: &[u8]) -> u64 {
    data.iter()
        .map(|&byte| (byte & 0x3F).count_ones() as u64)
        .sum()
}
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    #[test]
    fn test_dot_product_quaternary_vs_theoretical() {
        let mut rng = rand::thread_rng();
        let length = 32;
        // Generate random vectors with values in the range [0, 1, 2, 3]
        let input_a: Vec<u8> = (0..length).map(|_| rng.gen_range(0..=3)).collect();
        let input_b: Vec<u8> = (0..length).map(|_| rng.gen_range(0..=3)).collect();

        let mut vec_a = vec![vec![0u8; (length + 3) / 4]; 2];
        let mut vec_b = vec![vec![0u8; (length + 3) / 4]; 2];

        // Pack quaternary digits into u8 values
        for (i, (&a, &b)) in input_a.iter().zip(input_b.iter()).enumerate() {
            let byte_index = i / 4;
            let bit_offset = (i % 4) * 2;

            vec_a[0][byte_index] |= (a & 1) << bit_offset;
            vec_a[1][byte_index] |= ((a >> 1) & 1) << bit_offset;

            vec_b[0][byte_index] |= (b & 1) << bit_offset;
            vec_b[1][byte_index] |= ((b >> 1) & 1) << bit_offset;
        }

        // Compute dot product using quaternary method
        let quaternary_dot_product = dot_product_quaternary(&vec_a, &vec_b, 2);

        // Compute theoretical dot product
        let float_a: Vec<f32> = input_a.iter().map(|&x| x as f32).collect();
        let float_b: Vec<f32> = input_b.iter().map(|&x| x as f32).collect();
        let theoretical_dot_product = theoretical_dot_product(&float_a, &float_b);

        // Calculate relative error
        let relative_error = ((quaternary_dot_product - theoretical_dot_product).abs()
            / theoretical_dot_product.abs())
            * 100.0;

        // Assert that the relative error is within an acceptable range (e.g., 1%)
        assert!(
            relative_error < 1.0,
            "Relative error too high: {}%. Quaternary: {}, Theoretical: {}",
            relative_error,
            quaternary_dot_product,
            theoretical_dot_product
        );

        // Optionally, print detailed results for debugging
        println!("Input A: {:?}", input_a);
        println!("Input B: {:?}", input_b);
        println!("Quaternary dot product: {}", quaternary_dot_product);
        println!("Theoretical dot product: {}", theoretical_dot_product);
        println!("Relative error: {}%", relative_error);
    }

    fn theoretical_dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn generate_random_vectors(size: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut rng = rand::thread_rng();
        let mut generate_random_u8_vec = || (0..size).map(|_| rng.gen::<u8>()).collect();
        let x_vec = vec![generate_random_u8_vec(), generate_random_u8_vec()];
        let y_vec = vec![generate_random_u8_vec(), generate_random_u8_vec()];
        (x_vec, y_vec)
    }

    use std::time::Instant;
    #[test]
    fn test_dot_product_quaternary_correctness() {
        let sizes = vec![128, 256, 512, 1024];

        for size in sizes {
            let (x_vec, y_vec) = generate_random_vectors(size);

            let non_simd_result = dot_product_quaternary(&x_vec, &y_vec, 2);

            #[cfg(target_arch = "x86_64")]
            let simd_result = unsafe {
                if is_x86_feature_detected!("avx2") {
                    dot_product_quaternary_avx2(&x_vec, &y_vec, 2)
                } else {
                    non_simd_result // Fallback if AVX2 is not available
                }
            };

            #[cfg(not(target_arch = "x86_64"))]
            let simd_result = non_simd_result;

            let diff = (simd_result - non_simd_result).abs();
            const EPSILON: f32 = 1e-6;

            assert!(
                diff < EPSILON,
                "Results don't match for size {}: SIMD = {}, Non-SIMD = {}",
                size,
                simd_result,
                non_simd_result
            );
        }
    }

    #[test]
    fn test_dot_product_quaternary_performance() {
        let sizes = vec![128, 256, 512, 1024, 2048, 4096];
        let num_tests = 100;

        for size in sizes {
            let mut simd_time = 0.0;
            let mut non_simd_time = 0.0;

            for _ in 0..num_tests {
                let (x_vec, y_vec) = generate_random_vectors(size);

                // Non-SIMD version
                let start = Instant::now();
                let _ = dot_product_quaternary(&x_vec, &y_vec, 2);
                non_simd_time += start.elapsed().as_secs_f64();

                // SIMD version
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if is_x86_feature_detected!("avx2") {
                        let start = Instant::now();
                        let _ = dot_product_quaternary_avx2(&x_vec, &y_vec, 2);
                        simd_time += start.elapsed().as_secs_f64();
                    } else {
                        simd_time = non_simd_time; // Fallback if AVX2 is not available
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    simd_time = non_simd_time;
                }
            }

            println!("Size: {}", size);
            println!(
                "Average non-SIMD time: {} seconds",
                non_simd_time / num_tests as f64
            );
            println!(
                "Average SIMD time: {} seconds",
                simd_time / num_tests as f64
            );
            println!("Speedup: {:.2}x", non_simd_time / simd_time);
            println!();
        }
    }

    // Scalar equivalent function
    fn count_ones_scalar(input: u32) -> u32 {
        input.count_ones()
    }

    // Helper function to convert __m256i to Vec<u32>
    unsafe fn m256i_to_vec(v: __m256i) -> Vec<u32> {
        let mut result = vec![0u32; 8];
        _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, v);
        result
    }

    // Helper function to create __m256i from Vec<u32>
    unsafe fn vec_to_m256i(v: &[u32]) -> __m256i {
        assert!(v.len() >= 8);
        _mm256_loadu_si256(v.as_ptr() as *const __m256i)
    }

    #[test]
    fn test_count_ones_simple_cases() {
        unsafe {
            let test_cases = vec![
                vec![0u32; 8],
                vec![1u32; 8],
                vec![0xFFFFFFFF; 8],
                vec![0x55555555; 8],
                vec![0xAAAAAAAA; 8],
            ];

            for case in test_cases {
                let input = vec_to_m256i(&case);
                let avx2_result = count_ones_simd_avx2_256i(input);
                let scalar_result: u64 = case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
                assert_eq!(avx2_result, scalar_result, "Failed for case: {:?}", case);
            }
        }
    }

    #[test]
    fn test_count_ones_random_cases() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        unsafe {
            for _ in 0..100 {
                let case: Vec<u32> = (0..8).map(|_| rng.gen()).collect();
                let input = vec_to_m256i(&case);
                let avx2_result = count_ones_simd_avx2_256i(input);
                let scalar_result: u64 = case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
                assert_eq!(avx2_result, scalar_result, "Failed for case: {:?}", case);
            }
        }
    }

    #[test]
    fn test_count_ones_edge_cases() {
        unsafe {
            let edge_cases = vec![
                vec![
                    0u32, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF,
                ],
                vec![
                    0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x76543210, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF,
                ],
                vec![
                    0x01010101, 0x10101010, 0x11111111, 0xEEEEEEEE, 0xFFFF0000, 0x0000FFFF,
                    0xF0F0F0F0, 0x0F0F0F0F,
                ],
            ];

            for case in edge_cases {
                let input = vec_to_m256i(&case);
                let avx2_result = count_ones_simd_avx2_256i(input);
                let scalar_result: u64 = case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
                assert_eq!(avx2_result, scalar_result, "Failed for case: {:?}", case);
            }
        }
    }

    #[test]
    fn test_count_ones_incremental() {
        unsafe {
            let mut case = vec![0u32; 8];
            for i in 0..256 {
                case[i / 32] |= 1 << (i % 32);
                let input = vec_to_m256i(&case);
                let avx2_result = count_ones_simd_avx2_256i(input);
                let scalar_result: u64 = case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
                assert_eq!(avx2_result, scalar_result, "Failed for case: {:?}", case);
            }
        }
    }

    #[test]
    fn test_combinations() {
        let sizes = [100, 1000, 10000];

        for &size in &sizes {
            let data: Vec<u8> = (0..size).map(|i| i as u8).collect();

            // Warm-up
            let _ = quaternary_weighted_wrapper(&data);
            let _ = scalar_combinations(&data);

            // AVX2 version
            let start = Instant::now();
            let avx2_result = quaternary_weighted_wrapper(&data);
            let avx2_duration = start.elapsed();

            // Scalar version
            let start = Instant::now();
            let scalar_result = scalar_combinations(&data);
            let scalar_duration = start.elapsed();

            println!("Size: {}", size);
            println!("AVX2 result: {}, time: {:?}", avx2_result, avx2_duration);
            println!(
                "Scalar result: {}, time: {:?}",
                scalar_result, scalar_duration
            );
            println!("Results match: {}", avx2_result == scalar_result);
            println!(
                "Speedup: {:.2}x",
                scalar_duration.as_secs_f64() / avx2_duration.as_secs_f64()
            );
            println!();
        }
    }

    #[test]
    fn test_octal_weighted_popcount() {
        let mut rng = rand::thread_rng();

        // Test with various sizes
        for &size in &[32, 33, 64, 1024, 10000, 100000] {
            // Generate random data
            let data: Vec<u8> = (0..size).map(|_| rng.gen()).collect();

            // Run both implementations
            let octal_result = octal_weighted_wrapper(&data);
            let scalar_result = scalar_u6_count_ones(&data);

            // Assert that results match
            assert_eq!(octal_result, scalar_result, "Mismatch for size {}", size);
            println!("Test passed for size {}", size);
        }

        // Test with edge cases
        let edge_cases = vec![
            vec![0u8; 32],                     // All zeros
            vec![255u8; 64],                   // All ones
            vec![0, 255, 0, 255, 0, 255],      // Alternating zeros and ones
            vec![1, 2, 4, 8, 16, 32, 64, 128], // Powers of 2
            vec![85, 170, 85, 170],            // Alternating bit patterns
        ];

        for (i, case) in edge_cases.iter().enumerate() {
            let octal_result = octal_weighted_wrapper(case);
            let scalar_result = scalar_u6_count_ones(case);

            assert_eq!(octal_result, scalar_result, "Mismatch for edge case {}", i);
            println!("Edge case {} passed", i);
        }

        println!("All tests passed!");
    }
}
#[cfg(target_arch = "x86_64")]
fn scalar_combinations(data: &[u8]) -> u64 {
    data.iter().map(|&byte| byte.count_ones() as u64).sum()
}
#[cfg(target_arch = "x86_64")]
fn generate_test_vectors(size: usize, pattern: u32) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let lsb = vec![pattern; size];
    let msb = vec![pattern; size];
    (vec![lsb.clone(), msb.clone()], vec![lsb, msb])
}

#[cfg(target_arch = "x86_64")]
fn count_combinations_scalar(a_vec: &[Vec<u8>], b_vec: &[Vec<u8>]) -> [u64; 16] {
    let mut counts = [0u64; 16];

    // Ensure vectors are not empty and have the same length
    assert!(a_vec.len() == 2 && b_vec.len() == 2);
    let len = a_vec[0].len();
    assert!(len == a_vec[1].len() && len == b_vec[0].len() && len == b_vec[1].len());

    // Process each element in the vectors
    for i in 0..len {
        // Process each bit position in the u8 values
        for bit_pos in 0..8 {
            // Process each bit from 0 to 7
            let bit_mask = 1 << bit_pos;

            // Extract bits for this position
            let a_lsb = (a_vec[0][i] & bit_mask) >> bit_pos;
            let a_msb = ((a_vec[1][i] & bit_mask) >> bit_pos) << 1;
            let b_lsb = ((b_vec[0][i] & bit_mask) >> bit_pos) << 2;
            let b_msb = ((b_vec[1][i] & bit_mask) >> bit_pos) << 3;

            // Compute the 4-bit index
            let index = (a_msb | a_lsb | b_lsb | b_msb) as usize;

            // Ensure the index is within bounds
            if index < 16 {
                counts[index] += 1;
            } else {
                eprintln!("Index out of bounds: {}", index);
            }
        }
    }

    counts
}
