use rand::Rng;
use std::arch::x86_64::*;

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

pub fn dot_product_f32_chunk(src: &[(f32, f32)], dst: &mut [f32]) -> f32 {
    let mut d: f32 = 0.0;
    let chunk_size = 4;

    // Process chunks of 4
    for chunk in src.chunks_exact(chunk_size) {
        let mut local_sum: f32 = 0.0;
        local_sum += chunk[0].0 * chunk[0].1;
        local_sum += chunk[1].0 * chunk[1].1;
        local_sum += chunk[2].0 * chunk[2].1;
        local_sum += chunk[3].0 * chunk[3].1;
        d += local_sum;
    }

    // Handle remaining elements
    for &(a, b) in src.chunks_exact(chunk_size).remainder() {
        d += a * b;
    }

    d
}
pub fn dot_product_u8_chunk(src: &[(u8, u8)]) -> u64 {
    let mut d: u64 = 0;
    let chunk_size = 8;

    // Process chunks of 8
    for chunk in src.chunks_exact(chunk_size) {
        let mut local_sum: u64 = 0;
        local_sum += (chunk[0].0 as u64) * (chunk[0].1 as u64);
        local_sum += (chunk[1].0 as u64) * (chunk[1].1 as u64);
        local_sum += (chunk[2].0 as u64) * (chunk[2].1 as u64);
        local_sum += (chunk[3].0 as u64) * (chunk[3].1 as u64);
        local_sum += (chunk[4].0 as u64) * (chunk[4].1 as u64);
        local_sum += (chunk[5].0 as u64) * (chunk[5].1 as u64);
        local_sum += (chunk[6].0 as u64) * (chunk[6].1 as u64);
        local_sum += (chunk[7].0 as u64) * (chunk[7].1 as u64);
        d += local_sum;
    }

    // Handle remaining elements
    for &(a, b) in src.chunks_exact(chunk_size).remainder() {
        d += (a as u64) * (b as u64);
    }

    d
}
pub fn dot_product_a(src: &[(f32, f32)], dst: &mut [f32]) -> f32 {
    let mut d: f32 = 0.0;
    for (dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
        d += (src_sample.0 * src_sample.1);
    }
    d
}

pub fn dot_product_b(src: &[(f32, f32)], dst: &mut [f32]) {
    for (dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
        *dst_sample = (src_sample.0 * src_sample.1);
    }
}

pub fn dot_product_u8(src: &[(u8, u8)]) -> u64 {
    src.iter().map(|&(a, b)| (a as u64) * (b as u64)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    #[test]
    fn test_dot_product_functions() {
        let sizes = [32, 64, 128, 1024, 2048]; //, 64, 100, 128, 256, 500, 1000, 1024, 2048];

        for &size in &sizes {
            let mut rng = rand::thread_rng();

            // Generate data for u8 tests
            //let a_u8: Vec<u8> = (0..size).map(|_| rng.gen()).collect();
            //let b_u8: Vec<u8> = (0..size).map(|_| rng.gen()).collect();
            let a_u8: Vec<u8> = (0..size).map(|_| rng.gen_range(0..15)).collect();
            let b_u8: Vec<u8> = (0..size).map(|_| rng.gen_range(0..15)).collect();
            let ab_u8: Vec<(u8, u8)> = a_u8
                .iter()
                .zip(b_u8.iter())
                .map(|(&a, &b)| (a, b))
                .collect();
            // Pack two consecutive `u8` values into a single `u8`
            let packed_a_u8: Vec<u8> = a_u8
                .chunks(2)
                .map(|chunk| (chunk[0] & 0x0F) | ((chunk[1] & 0x0F) << 4))
                .collect();
            // Pack two consecutive `u8` values into a single `u8`
            let packed_b_u8: Vec<u8> = b_u8
                .chunks(2)
                .map(|chunk| (chunk[0] & 0x0F) | ((chunk[1] & 0x0F) << 4))
                .collect();
            // Generate data for f32 tests
            let a_f32: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
            let b_f32: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
            let ab_f32: Vec<(f32, f32)> = a_f32
                .iter()
                .zip(b_f32.iter())
                .map(|(&a, &b)| (a, b))
                .collect();
            let mut dst_f32 = vec![0.0; size];

            // Test u8 functions
            let result_avx2 = unsafe { dot_product_u8_avx2(&a_u8, &b_u8) };
            let result_u8_chunk = dot_product_u8_chunk(&ab_u8);
            let result_u8_scalar = dot_product_u8(&ab_u8);
            println!("Size: {}", size);
            println!("u8 results:");

            println!("  AVX2:       {}", result_avx2);
            println!("  Chunk:      {}", result_u8_chunk);
            println!("  Scalar:     {}", result_u8_scalar);
            // assert_eq!(
            //     result_fma, result_avx2,
            //     "Results differ for u8 AVX2 FMA vs AVX2 for size {}",
            //     size
            // );
            // assert_eq!(
            //     result_fma, result_u8_chunk,
            //     "Results differ for u8 AVX2 FMA vs chunk for size {}",
            //     size
            // );
            // assert_eq!(
            //     result_fma, result_u8_scalar,
            //     "Results differ for u8 AVX2 FMA vs scalar for size {}",
            //     size
            // );

            // Test f32 functions
            let result_f32_chunk = dot_product_f32_chunk(&ab_f32, &mut dst_f32);
            let result_f32_a = dot_product_a(&ab_f32, &mut dst_f32);
            dot_product_b(&ab_f32, &mut dst_f32);
            let result_f32_b: f32 = dst_f32.iter().sum();

            let scalar_result_f32: f32 = ab_f32.iter().map(|&(a, b)| a * b).sum();
            println!("f32 results:");
            println!("  Chunk:      {}", result_f32_chunk);
            println!("  A:          {}", result_f32_a);
            println!("  B:          {}", result_f32_b);
            println!("  Scalar:     {}", scalar_result_f32);
            println!();
            // assert!(
            //     (result_f32_chunk - scalar_result_f32).abs() < 1e-6,
            //     "Results differ for f32 chunk vs scalar for size {}",
            //     size
            // );
            // assert!(
            //     (result_f32_a - scalar_result_f32).abs() < 1e-6,
            //     "Results differ for f32 A vs scalar for size {}",
            //     size
            // );
            // assert!(
            //     (result_f32_b - scalar_result_f32).abs() < 1e-6,
            //     "Results differ for f32 B vs scalar for size {}",
            //     size
            // );
        }
    }
}
