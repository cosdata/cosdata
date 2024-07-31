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

    let mut dot_product = 0u32;

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

        let lsbs_count = _mm256_popcnt_epi32(lsbs);
        let carry_count = _mm256_popcnt_epi32(carry);
        let msbs_count = _mm256_popcnt_epi32(msbs);
        let mid_count = _mm256_popcnt_epi32(mid);

        let msbs_shifted = _mm256_slli_epi32(msbs_count, 2);
        let carry_shifted = _mm256_slli_epi32(carry_count, 2);
        let mid_shifted = _mm256_slli_epi32(mid_count, 1);

        let result = _mm256_add_epi32(
            _mm256_add_epi32(msbs_shifted, carry_shifted),
            _mm256_add_epi32(mid_shifted, lsbs_count),
        );

        let result_array = std::mem::transmute::<__m256i, [u32; 8]>(result);
        dot_product += result_array.iter().sum::<u32>();
    }

    dot_product as f32
}
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
}
