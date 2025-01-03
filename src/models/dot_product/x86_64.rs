use std::arch::x86_64::*;

#[allow(dead_code)]
fn print_mm256i(name: &str, value: __m256i) {
    let mut array = [0u8; 32];
    unsafe {
        _mm256_storeu_si256(array.as_mut_ptr() as *mut __m256i, value);
    }
    print!("{} = [", name);
    for (i, &byte) in array.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:3}", byte);
    }
    println!("]");
}

#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_u8_avx2(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());

    let mut dot_product: u64 = 0;
    let len = a.len();
    let chunk_size = 32;

    let mut sumlo = _mm256_setzero_si256();
    let mut sumhi = _mm256_setzero_si256();
    // Process 32 elements at a time
    let mut i = 0;
    while i + chunk_size <= len {
        // Load 32 elements from each array into AVX2 registers
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        // Unpack 8-bit integers to 16-bit integers
        let va_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        let va_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        let vb_lo = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        let vb_hi = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        // print_mm256i("va_lo", va_lo);

        // Multiply and add packed 16-bit integers
        let prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
        let prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
        // print_mm256i("prod_lo", prod_lo);

        sumlo = _mm256_add_epi32(sumlo, prod_lo);
        sumhi = _mm256_add_epi32(sumhi, prod_hi);

        i += chunk_size;
    }

    dot_product += accumulate_u32(sumlo) as u64;
    dot_product += accumulate_u32(sumhi) as u64;

    // Handle remaining elements
    while i < len {
        dot_product += a[i] as u64 * b[i] as u64;
        i += 1;
    }
    dot_product
}

#[target_feature(enable = "avx2")]
unsafe fn accumulate_u32(x: __m256i) -> u32 {
    // Horizontal add within 256-bit lanes
    let sum1 = _mm256_hadd_epi32(x, x);
    // Horizontal add again
    let sum2 = _mm256_hadd_epi32(sum1, sum1);
    // Extract lower 128 bits
    let sum_low = _mm256_castsi256_si128(sum2);
    // Extract upper 128 bits
    let sum_high = _mm256_extracti128_si256::<1>(sum2);
    // Add lower and upper 128-bit results
    let result = _mm_add_epi32(sum_low, sum_high);
    // Extract the final sum
    _mm_cvtsi128_si32(result) as u32
}

#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_u64(x: __m256i) -> u64 {
    let zero = _mm256_setzero_si256();
    // Unpack to 64-bit integers
    let sum_low = _mm256_unpacklo_epi32(x, zero);
    let sum_high = _mm256_unpackhi_epi32(x, zero);
    // Add the low and high parts
    let sum = _mm256_add_epi64(sum_low, sum_high);
    // Extract lower and upper 128-bit lanes
    let sum_low = _mm256_extracti128_si256::<0>(sum);
    let sum_high = _mm256_extracti128_si256::<1>(sum);
    // Add the low and high 128-bit lanes
    let sum = _mm_add_epi64(sum_low, sum_high);
    // Extract and add the two 64-bit integers
    _mm_extract_epi64(sum, 0) as u64 + _mm_extract_epi64(sum, 1) as u64
}

#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_quaternary_avx2(
    x_vec: &[Vec<u8>],
    y_vec: &[Vec<u8>],
    resolution: u8,
) -> f32 {
    assert_eq!(resolution, 2);
    let len = x_vec[0].len();

    let mut dot_product = 0u64;

    let mut i = 0;

    while i + 32 < len {
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
        i += 32;
    }

    for i in i..len {
        let x_lsb = x_vec[0][i];
        let x_msb = x_vec[1][i];
        let y_lsb = y_vec[0][i];
        let y_msb = y_vec[1][i];
        let lsbs = (x_lsb & y_lsb).count_ones();
        let mid1 = x_lsb & y_msb;
        let mid2 = y_lsb & x_msb;
        let carry = (mid1 & mid2).count_ones();
        let msbs = (x_msb & y_msb).count_ones();
        let mid = (mid1 ^ mid2).count_ones();

        let result = (msbs << 2) + (carry << 2) + (mid << 1) + lsbs;
        dot_product += result as u64;
    }

    dot_product as f32
}

#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_binary_avx2(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], resolution: u8) -> f32 {
    assert_eq!(resolution, 1);

    let len = x_vec[0].len();

    let mut dot_product = 0u64;

    let mut i = 0;

    while i + 32 < len {
        let x = _mm256_loadu_si256(x_vec[0][i..].as_ptr() as *const __m256i);
        let y = _mm256_loadu_si256(y_vec[0][i..].as_ptr() as *const __m256i);
        let and = _mm256_and_si256(x, y);
        let count = count_ones_simd_avx2_256i(and);
        dot_product += count;

        i += 32;
    }

    for i in i..len {
        dot_product += (x_vec[0][i] & y_vec[0][i]).count_ones() as u64;
    }

    dot_product as f32
}

#[target_feature(enable = "avx2")]
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

#[allow(dead_code)]
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
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn quaternary_weighted_wrapper(data: &[u8]) -> u64 {
    let lookup: [u8; 32] = [
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, // repeated
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    ];
    quaternary_weighted_simd_avx2(data.as_ptr(), data.len(), &lookup)
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

#[target_feature(enable = "avx2")]
pub unsafe fn octal_weighted_wrapper(data: &[u8]) -> u64 {
    // Initialize lookup table with a * b for a, b in 0..8
    #[rustfmt::skip]
    let lookup = [
        // MSB = 00
        0, 0, 0, 0,
        0, 1, 2, 3,
        0, 2, 4, 6,
        0, 3, 6, 9,

        // MSB = 01
        0, 0, 0, 0,
        4, 5, 6, 7,
        8, 10, 12, 14,
        12, 15, 18, 21,

        // MSB = 10
        0, 4, 8, 12,
        0, 5, 10, 15,
        0, 6, 12, 18,
        0, 7, 14, 21,

        // MSB = 11
        16, 20, 24, 28,
        20, 25, 30, 35,
        24, 30, 36, 42,
        28, 35, 42, 49,
    ];

    unsafe { octal_weighted_simd_avx2(data.as_ptr(), data.len(), &lookup) }
}

#[target_feature(enable = "avx2")]
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
        for _ in 0..2 {
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

macro_rules! pack_octal {
    ($x_vec: ident, $y_vec: ident, $mask:ident, $i: ident, $j: literal) => {{
        (($y_vec[0][$i] & $mask) >> $j)
            | ((($y_vec[1][$i] & $mask) >> $j) << 1)
            | ((($x_vec[0][$i] & $mask) >> $j) << 2)
            | ((($x_vec[1][$i] & $mask) >> $j) << 3)
            | ((($y_vec[2][$i] & $mask) >> $j) << 4)
            | ((($x_vec[2][$i] & $mask) >> $j) << 5)
    }};
}

pub fn pack_octal_vectors(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>]) -> Vec<u8> {
    const MASK_0: u8 = 1 << 0;
    const MASK_1: u8 = 1 << 1;
    const MASK_2: u8 = 1 << 2;
    const MASK_3: u8 = 1 << 3;
    const MASK_4: u8 = 1 << 4;
    const MASK_5: u8 = 1 << 5;
    const MASK_6: u8 = 1 << 6;
    const MASK_7: u8 = 1 << 7;

    let mut data = Vec::with_capacity(x_vec[0].len() * 8);
    for i in 0..x_vec[0].len() {
        data.push(pack_octal!(x_vec, y_vec, MASK_0, i, 0));
        data.push(pack_octal!(x_vec, y_vec, MASK_1, i, 1));
        data.push(pack_octal!(x_vec, y_vec, MASK_2, i, 2));
        data.push(pack_octal!(x_vec, y_vec, MASK_3, i, 3));
        data.push(pack_octal!(x_vec, y_vec, MASK_4, i, 4));
        data.push(pack_octal!(x_vec, y_vec, MASK_5, i, 5));
        data.push(pack_octal!(x_vec, y_vec, MASK_6, i, 6));
        data.push(pack_octal!(x_vec, y_vec, MASK_7, i, 7));
    }
    data
}

// Scalar implementation for comparison
#[allow(dead_code)]
fn scalar_u6_count_ones(data: &[u8]) -> u64 {
    data.iter()
        .map(|&byte| (byte & 0x3F).count_ones() as u64)
        .sum()
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_product_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");

    let n = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a[offset..].as_ptr());
        let vb = _mm256_loadu_ps(b[offset..].as_ptr());
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let temp = _mm256_hadd_ps(sum, sum);
    let temp = _mm256_hadd_ps(temp, temp);
    let sum_low = _mm256_castps256_ps128(temp);
    let sum_high = _mm256_extractf128_ps(temp, 1);
    let final_sum = _mm_add_ps(sum_low, sum_high);

    let mut result = _mm_cvtss_f32(final_sum);
    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::dot_product::{
        dot_product_binary_scalar, dot_product_octal_scalar, dot_product_quaternary_scalar,
    };
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
        let quaternary_dot_product = dot_product_quaternary_scalar(&vec_a, &vec_b, 2);

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

    fn generate_random_octal_vectors(size: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut rng = rand::thread_rng();
        let mut generate_random_u8_vec = || (0..size).map(|_| rng.gen::<u8>()).collect();
        let x_vec = vec![
            generate_random_u8_vec(),
            generate_random_u8_vec(),
            generate_random_u8_vec(),
        ];
        let y_vec = vec![
            generate_random_u8_vec(),
            generate_random_u8_vec(),
            generate_random_u8_vec(),
        ];
        (x_vec, y_vec)
    }

    fn generate_random_quaternary_vectors(size: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut rng = rand::thread_rng();
        let mut generate_random_u8_vec = || (0..size).map(|_| rng.gen::<u8>()).collect();
        let x_vec = vec![generate_random_u8_vec(), generate_random_u8_vec()];
        let y_vec = vec![generate_random_u8_vec(), generate_random_u8_vec()];
        (x_vec, y_vec)
    }

    fn generate_random_binary_vectors(size: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut rng = rand::thread_rng();
        let mut generate_random_u8_vec = || (0..size).map(|_| rng.gen::<u8>()).collect();
        let x_vec = vec![generate_random_u8_vec()];
        let y_vec = vec![generate_random_u8_vec()];
        (x_vec, y_vec)
    }

    use std::time::Instant;
    #[test]
    fn test_dot_product_quaternary_correctness() {
        let sizes = vec![128, 256, 512, 1024];

        for size in sizes {
            let (x_vec, y_vec) = generate_random_quaternary_vectors(size);

            let non_simd_result = dot_product_quaternary_scalar(&x_vec, &y_vec, 2);

            let simd_result = unsafe {
                if is_x86_feature_detected!("avx2") {
                    dot_product_quaternary_avx2(&x_vec, &y_vec, 2)
                } else {
                    non_simd_result // Fallback if AVX2 is not available
                }
            };

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
    fn test_dot_product_binary_correctness() {
        let sizes = vec![128, 256, 512, 1024];

        for size in sizes {
            let (x_vec, y_vec) = generate_random_binary_vectors(size);

            let non_simd_result = dot_product_binary_scalar(&x_vec, &y_vec, 1);

            let simd_result = unsafe {
                if is_x86_feature_detected!("avx2") {
                    dot_product_binary_avx2(&x_vec, &y_vec, 1)
                } else {
                    non_simd_result // Fallback if AVX2 is not available
                }
            };

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
                let (x_vec, y_vec) = generate_random_quaternary_vectors(size);

                // Non-SIMD version
                let start = Instant::now();
                let _ = dot_product_quaternary_scalar(&x_vec, &y_vec, 2);
                non_simd_time += start.elapsed().as_secs_f64();

                // SIMD version
                #[allow(unused_assignments)]
                unsafe {
                    if is_x86_feature_detected!("avx2") {
                        let start = Instant::now();
                        let _ = dot_product_quaternary_avx2(&x_vec, &y_vec, 2);
                        simd_time += start.elapsed().as_secs_f64();
                    } else {
                        simd_time = non_simd_time; // Fallback if AVX2 is not available
                    }
                }

                simd_time = non_simd_time;
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
    #[allow(dead_code)]
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
            let _ = unsafe { quaternary_weighted_wrapper(&data) };
            let _ = scalar_combinations(&data);

            // AVX2 version
            let start = Instant::now();
            let avx2_result = unsafe { quaternary_weighted_wrapper(&data) };
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
    fn test_dot_product_octal_correctness() {
        let sizes = vec![64, 128, 256, 512, 1024];

        for size in sizes {
            let (x_vec, y_vec) = generate_random_octal_vectors(size);

            let scalar_result = dot_product_octal_scalar(&x_vec, &y_vec, 3);
            let packed = pack_octal_vectors(&x_vec, &y_vec);

            let simd_result = unsafe {
                if is_x86_feature_detected!("avx2") {
                    octal_weighted_wrapper(&packed) as f32
                } else {
                    0.0
                }
            };

            let diff = (simd_result - scalar_result).abs();
            const EPSILON: f32 = 1e-6;

            println!("Final SIMD dot product: {}", simd_result);
            println!("Final Scalar dot product: {}", scalar_result);

            assert!(
                diff < EPSILON,
                "Results don't match for size {}: SIMD = {}, Non-SIMD = {}",
                size,
                simd_result,
                scalar_result,
            );
        }
    }
}

#[allow(dead_code)]
fn scalar_combinations(data: &[u8]) -> u64 {
    data.iter().map(|&byte| byte.count_ones() as u64).sum()
}

#[allow(dead_code)]
fn generate_test_vectors(size: usize, pattern: u32) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let lsb = vec![pattern; size];
    let msb = vec![pattern; size];
    (vec![lsb.clone(), msb.clone()], vec![lsb, msb])
}

#[allow(dead_code)]
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
