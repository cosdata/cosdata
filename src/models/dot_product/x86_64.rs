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
