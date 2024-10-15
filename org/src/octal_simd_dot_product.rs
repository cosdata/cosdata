use rand::{thread_rng, Rng};
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn octal_weighted_wrapper(data: &[u8]) -> u64 {
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
        //println!("outer loop: {}", i);
        for _ in 0..2 {
            if i + 32 >= n {
                break;
            }
            //println!("simd loop: {}", i);
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

    println!("interim SIMD result: {}", result);
    println!("handle remaining with scalar...");
    // Handle remaining bytes
    while i < n {
        let byte = *data.add(i) & 0x3F; // Mask to lower 6 bits
        let tr = lookup[byte as usize] as u64;
        //println!("i: {}, tr: {}", i, tr);
        result += tr;
        i += 1;
    }

    result
}

pub fn pack_octal_vectors(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>]) -> Vec<u8> {
    let mut data = Vec::with_capacity(x_vec[0].len() * 8);
    for i in 0..x_vec[0].len() {
        for j in 0..8 {
            let mask = 1u8 << j;
            let num = ((y_vec[0][i] & mask) >> j)
                | (((y_vec[1][i] & mask) >> j) << 1)
                | (((x_vec[0][i] & mask) >> j) << 2)
                | (((x_vec[1][i] & mask) >> j) << 3)
                | (((y_vec[2][i] & mask) >> j) << 4)
                | (((x_vec[2][i] & mask) >> j) << 5);
            data.push(num);
        }
    }
    data
}

fn dot_product_octal_scalar(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], res: u8) -> f32 {
    debug_assert_eq!(x_vec.len(), 3);
    debug_assert_eq!(y_vec.len(), 3);
    debug_assert_eq!(res, 3);
    let dot_product: u32 = x_vec[0]
        .iter()
        .zip(&x_vec[1])
        .zip(&x_vec[2])
        .zip(y_vec[0].iter().zip(&y_vec[1]).zip(&y_vec[2]))
        .map(|(((&x_lsb, &x_mid), &x_msb), ((&y_lsb, &y_mid), &y_msb))| {
            let mut sum = 0u32;
            for bit in 0..8 {
                let x = ((x_msb & (1 << bit)) >> bit) << 2
                    | ((x_mid & (1 << bit)) >> bit) << 1
                    | ((x_lsb & (1 << bit)) >> bit);
                let y = ((y_msb & (1 << bit)) >> bit) << 2
                    | ((y_mid & (1 << bit)) >> bit) << 1
                    | ((y_lsb & (1 << bit)) >> bit);
                sum += (x * y) as u32;
            }
            sum
        })
        .sum();
    dot_product as f32
}

fn generate_random_octal_vectors(size: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut rng = thread_rng();
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

fn main() {
    let sizes = vec![64, 128, 256, 512, 1024];

    for size in sizes {
        let (x_vec, y_vec) = generate_random_octal_vectors(size);

        println!("\n\nX,Y vec length: {:?}", x_vec[0].len());

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
