use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

#[cfg(target_arch = "x86_64")]
pub fn dot_product_u8_avx2_fma(a: &[u8], b: &[u8]) -> u64 {
    use std::arch::x86_64::*;
    assert_eq!(a.len(), b.len());

    let mut dot_product: u64 = 0;

    // Process 32 elements at a time
    let mut i = 0;
    while i + 32 <= a.len() {
        unsafe {
            // Load 32 elements from each array into AVX2 registers
            let va1 = _mm256_loadu_si256(a[i..].as_ptr() as *const __m256i);
            let vb1 = _mm256_loadu_si256(b[i..].as_ptr() as *const __m256i);

            // Unpack to 16-bit integers
            let va1_lo = _mm256_unpacklo_epi8(va1, _mm256_setzero_si256());
            let vb1_lo = _mm256_unpacklo_epi8(vb1, _mm256_setzero_si256());
            let prod1_lo = _mm256_madd_epi16(va1_lo, vb1_lo);

            let va1_hi = _mm256_unpackhi_epi8(va1, _mm256_setzero_si256());
            let vb1_hi = _mm256_unpackhi_epi8(vb1, _mm256_setzero_si256());
            let prod1_hi = _mm256_madd_epi16(va1_hi, vb1_hi);

            // Horizontal add within 256-bit registers
            let sum1 = _mm256_add_epi32(prod1_lo, prod1_hi);
            let sum2 = _mm256_permute4x64_epi64(sum1, 0b11011000); // permute for horizontal add
            let sum3 = _mm256_hadd_epi32(sum2, sum2);
            let sum4 = _mm256_hadd_epi32(sum3, sum3);

            // Extract result to scalar
            dot_product += _mm256_extract_epi64(sum4, 0) as u64;
        }
        i += 32;
    }

    // Handle remaining elements
    while i < a.len() {
        dot_product += a[i] as u64 * b[i] as u64;
        i += 1;
    }

    dot_product
}

pub fn dot_product_a(src: &[(f32, f32)]) -> f32 {
    let mut d: f32 = 0.0;
    for src_sample in src.iter() {
        d += src_sample.0 * src_sample.1;
    }
    d
}

pub fn dot_product_b(src: &[(f32, f32)], dst: &mut [f32]) {
    for (dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
        *dst_sample = src_sample.0 * src_sample.1;
    }
}

pub fn dot_product_u8(src: &[(u8, u8)]) -> u64 {
    src.iter().map(|&(a, b)| (a as u64) * (b as u64)).sum()
}

fn generate_random_data(size: usize) -> (Vec<(f32, f32)>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let src: Vec<(f32, f32)> = (0..size)
        .map(|_| (rng.gen::<f32>(), rng.gen::<f32>()))
        .collect();
    let dst = vec![0.0; size];
    (src, dst)
}

fn generate_random_data_u8(size: usize) -> Vec<(u8, u8)> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| (rng.gen::<u8>(), rng.gen::<u8>()))
        .collect()
}

// Function to generate random u8 vectors
fn generate_random_vectors(count: usize, len: usize) -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..len).map(|_| rng.gen::<u8>()).collect())
        .collect()
}

fn bench_dot_product(c: &mut Criterion) {
    let sizes = [1000, 1500, 2000];
    let vector_count = 100;
    let _vector_length = 1024; // Make sure this is a multiple of your SIMD width

    for size in sizes.iter() {
        let (src, dst_a) = generate_random_data(*size);
        let mut dst_b = dst_a.clone();
        let src_u8 = generate_random_data_u8(*size);
        // Generate random vectors outside the benchmark loop
        let vectors = generate_random_vectors(vector_count, *size);

        c.bench_function(&format!("dot_product_a_{}", size), |b| {
            b.iter(|| {
                let result = dot_product_a(black_box(&src));
                black_box(result)
            })
        });

        c.bench_function(&format!("dot_product_b_{}", size), |b| {
            b.iter(|| {
                dot_product_b(black_box(&src), black_box(&mut dst_b));
                let sum: f32 = dst_b.iter().sum();
                black_box(sum)
            })
        });

        c.bench_function(&format!("dot_product_u8_{}", size), |b| {
            b.iter(|| {
                let result = dot_product_u8(black_box(&src_u8));
                black_box(result)
            })
        });

        #[cfg(target_arch = "x86_64")]
        c.bench_function(&format!("dot_product_u8_avx2_fma_{}", size), |b| {
            b.iter(|| {
                // Randomly select two vectors for each iteration
                let i = rand::thread_rng().gen_range(0..vector_count);
                let j = rand::thread_rng().gen_range(0..vector_count);
                let result =
                    dot_product_u8_avx2_fma(black_box(&vectors[i]), black_box(&vectors[j]));
                black_box(result)
            })
        });
    }
}

criterion_group!(benches, bench_dot_product);
criterion_main!(benches);
