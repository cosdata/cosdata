use super::{DistanceError, DistanceFunction};
use crate::models::lookup_table::shift_and_accumulate;
use crate::storage::Storage;
#[derive(Debug)]
pub struct CosineDistance;

impl DistanceFunction for CosineDistance {
    // Implementation here
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<f32, DistanceError> {
        match (x, y) {
            (
                Storage::UnsignedByte {
                    mag: mag_x,
                    quant_vec: vec_x,
                },
                Storage::UnsignedByte {
                    mag: mag_y,
                    quant_vec: vec_y,
                },
            ) => {
                // Implement cosine similarity for UnsignedByte storage
                unimplemented!("Cosine similarity for UnsignedByte not implemented yet")
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
    x_vec: &[Vec<u32>],
    y_vec: &[Vec<u32>],
    resolution: u8,
) -> Result<f32, DistanceError> {
    let parts = 2_usize.pow(resolution as u32);

    let mut final_result: usize = 0;

    for index in 0..parts {
        let sum: usize = x_vec[index]
            .iter()
            .zip(&y_vec[index])
            .map(|(&x_item, &y_item)| shift_and_accumulate(x_item & y_item) as usize)
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
) -> Result<f32, DistanceError> {
    let denominator = (mag_x as f32).sqrt() * (mag_y as f32).sqrt();
    if denominator == 0.0 {
        Err(DistanceError::CalculationError)
    } else {
        Ok(dot_product / denominator)
    }
}

fn dot_product_quaternary(x_vec: &[Vec<u32>], y_vec: &[Vec<u32>], resolution: u8) -> f32 {
    assert_eq!(resolution, 2);

    let dot_product: u32 = x_vec[0]
        .iter()
        .zip(&x_vec[1])
        .zip(y_vec[0].iter().zip(&y_vec[1]))
        .enumerate()
        .map(|(i, ((&x_lsb, &x_msb), (&y_lsb, &y_msb)))| {
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
unsafe fn dot_product_quaternary_avx2(
    x_vec: &[Vec<u32>],
    y_vec: &[Vec<u32>],
    resolution: u8,
) -> f32 {
    assert_eq!(resolution, 2);
    assert!(
        x_vec[0].len() % 8 == 0,
        "Vector length must be a multiple of 8"
    );

    let mut dot_product = 0u64;

    for i in (0..x_vec[0].len()).step_by(8) {
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
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    #[test]
    fn test_dot_product_quaternary_vs_theoretical() {
        let mut rng = rand::thread_rng();
        let length = 32;

        // Generate random vectors with values in the range [0, 1, 2, 3]
        let input_a: Vec<u32> = (0..length).map(|_| rng.gen_range(0..=3)).collect();
        let input_b: Vec<u32> = (0..length).map(|_| rng.gen_range(0..=3)).collect();

        let mut result_a = vec![vec![0; 32]; 2];
        let mut result_b = vec![vec![0; 32]; 2];

        // Process input_a
        for (i, &value) in input_a.iter().enumerate() {
            result_a[0][i] = value & 1;
            result_a[1][i] = (value >> 1) & 1;
        }

        // Process input_b
        for (i, &value) in input_b.iter().enumerate() {
            result_b[0][i] = value & 1;
            result_b[1][i] = (value >> 1) & 1;
        }

        let mut vec_a = vec![vec![0u32; 1]; 2];
        let mut vec_b = vec![vec![0u32; 1]; 2];

        // Combine bits for vec_a
        for (i, &bit) in result_a[0].iter().enumerate() {
            vec_a[0][0] |= (bit as u32) << i;
        }
        for (i, &bit) in result_a[1].iter().enumerate() {
            vec_a[1][0] |= (bit as u32) << i;
        }

        // Combine bits for vec_b
        for (i, &bit) in result_b[0].iter().enumerate() {
            vec_b[0][0] |= (bit as u32) << i;
        }
        for (i, &bit) in result_b[1].iter().enumerate() {
            vec_b[1][0] |= (bit as u32) << i;
        }

        // Compute dot product using binary method
        let binary_dot_product = dot_product_quaternary(&vec_a, &vec_b, 2);

        // Compute theoretical dot product
        let float_a: Vec<f32> = input_a.iter().map(|&x| x as f32).collect();
        let float_b: Vec<f32> = input_b.iter().map(|&x| x as f32).collect();
        let theoretical_dot_product = theoretical_dot_product(&float_a, &float_b);

        // Calculate relative error
        let relative_error = ((binary_dot_product - theoretical_dot_product).abs()
            / theoretical_dot_product.abs())
            * 100.0;

        // Assert that the relative error is within an acceptable range (e.g., 1%)
        assert!(
            relative_error < 1.0,
            "Relative error too high: {}%. Binary: {}, Theoretical: {}",
            relative_error,
            binary_dot_product,
            theoretical_dot_product
        );

        // Optionally, print detailed results for debugging
        println!("Input A: {:?}", input_a);
        println!("Input B: {:?}", input_b);
        println!("Quaternary dot product: {}", binary_dot_product);
        println!("Theoretical dot product: {}", theoretical_dot_product);
        println!("Relative error: {}%", relative_error);
    }

    fn theoretical_dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        use rand::Rng;
        use std::time::Instant;
        // Define the weight lookup table
        #[cfg(target_arch = "x86_64")]
        fn generate_random_vectors(size: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
            let mut rng = rand::thread_rng();
            let x_vec = vec![
                (0..size).map(|_| rng.gen::<u32>()).collect(),
                (0..size).map(|_| rng.gen::<u32>()).collect(),
            ];
            let y_vec = vec![
                (0..size).map(|_| rng.gen::<u32>()).collect(),
                (0..size).map(|_| rng.gen::<u32>()).collect(),
            ];
            (x_vec, y_vec)
        }
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
                    let scalar_result: u64 =
                        case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
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
                    let scalar_result: u64 =
                        case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
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
                        0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x76543210, 0, 0, 0xFFFFFFFF,
                        0xFFFFFFFF,
                    ],
                    vec![
                        0x01010101, 0x10101010, 0x11111111, 0xEEEEEEEE, 0xFFFF0000, 0x0000FFFF,
                        0xF0F0F0F0, 0x0F0F0F0F,
                    ],
                ];

                for case in edge_cases {
                    let input = vec_to_m256i(&case);
                    let avx2_result = count_ones_simd_avx2_256i(input);
                    let scalar_result: u64 =
                        case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
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
                    let scalar_result: u64 =
                        case.iter().map(|&x| count_ones_scalar(x) as u64).sum();
                    assert_eq!(avx2_result, scalar_result, "Failed for case: {:?}", case);
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn generate_test_vectors(size: usize, pattern: u32) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let lsb = vec![pattern; size];
        let msb = vec![pattern; size];
        (vec![lsb.clone(), msb.clone()], vec![lsb, msb])
    }

    #[cfg(target_arch = "x86_64")]
    fn count_combinations_scalar(a_vec: &[Vec<u32>], b_vec: &[Vec<u32>]) -> [u64; 16] {
        let mut counts = [0u64; 16];

        // Ensure vectors are not empty and have the same length
        assert!(a_vec.len() == 2 && b_vec.len() == 2);
        let len = a_vec[0].len();
        assert!(len == a_vec[1].len() && len == b_vec[0].len() && len == b_vec[1].len());

        // Process each element in the vectors
        for i in 0..len {
            // Process each bit position in the u32 values
            for bit_pos in 0..32 {
                // Process each bit from 0 to 31
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
}
