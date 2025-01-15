use half::f16;

#[cfg(target_arch = "aarch64")]
mod arm64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

fn dot_product_u8_scalar(a: &[u8], b: &[u8]) -> u64 {
    a.iter().zip(b).map(|(&x, &y)| x as u64 * y as u64).sum()
}

fn dot_product_f16_scalar(x_vec: &[f16], y_vec: &[f16]) -> f32 {
    x_vec
        .iter()
        .zip(y_vec)
        .map(|(&a, &b)| f32::from(a) * f32::from(b))
        .sum()
}

fn dot_product_binary_scalar(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], res: u8) -> f32 {
    debug_assert_eq!(x_vec.len(), 1);
    debug_assert_eq!(y_vec.len(), 1);
    debug_assert_eq!(res, 1);

    let dot_product: u32 = x_vec[0]
        .iter()
        .zip(&y_vec[0])
        .map(|(&x, &y)| (x & y).count_ones())
        .sum();

    dot_product as f32
}

fn dot_product_quaternary_scalar(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], res: u8) -> f32 {
    debug_assert_eq!(x_vec.len(), 2);
    debug_assert_eq!(y_vec.len(), 2);
    debug_assert_eq!(res, 2);

    let dot_product: u32 = x_vec[0]
        .iter()
        .zip(&x_vec[1])
        .zip(y_vec[0].iter().zip(&y_vec[1]))
        .map(|((&x_lsb, &x_msb), (&y_lsb, &y_msb))| {
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

fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
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

pub fn dot_product_u8(a: &[u8], b: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::dot_product_u8_avx2(a, b) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { arm64::dot_product_u8_neon(a, b) };
        }
    }

    dot_product_u8_scalar(a, b)
}

pub fn dot_product_f16(x: &[f16], y: &[f16]) -> f32 {
    // TODO: use SIMD if possible
    dot_product_f16_scalar(x, y)
}

pub fn dot_product_binary(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], res: u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::dot_product_binary_avx2(x_vec, y_vec, res) };
        }
    }
    dot_product_binary_scalar(x_vec, y_vec, res)
}

pub fn dot_product_quaternary(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], res: u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::dot_product_quaternary_avx2(x_vec, y_vec, res) };
        }
    }
    dot_product_quaternary_scalar(x_vec, y_vec, res)
}

pub fn dot_product_octal(x_vec: &[Vec<u8>], y_vec: &[Vec<u8>], res: u8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            let packed = x86_64::pack_octal_vectors(x_vec, y_vec);
            return unsafe { x86_64::octal_weighted_wrapper(&packed) as f32 };
        }
    }
    dot_product_octal_scalar(x_vec, y_vec, res)
}

pub fn dot_product_f32(x_vec: &[f32], y_vec: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { x86_64::dot_product_f32_simd(x_vec, y_vec) };
        }
    }
    dot_product_f32_scalar(x_vec, y_vec)
}

#[allow(dead_code)]
pub fn dot_product_f32_chunk(src: &[(f32, f32)], _dst: &mut [f32]) -> f32 {
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

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn dot_product_a(src: &[(f32, f32)], dst: &mut [f32]) -> f32 {
    let mut d: f32 = 0.0;
    for (_dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
        d += src_sample.0 * src_sample.1;
    }
    d
}

#[allow(dead_code)]
pub fn dot_product_b(src: &[(f32, f32)], dst: &mut [f32]) {
    for (dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
        *dst_sample = src_sample.0 * src_sample.1;
    }
}

#[allow(dead_code)]
pub fn dot_product_u8_zipped(src: &[(u8, u8)]) -> u64 {
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
            let _packed_a_u8: Vec<u8> = a_u8
                .chunks(2)
                .map(|chunk| (chunk[0] & 0x0F) | ((chunk[1] & 0x0F) << 4))
                .collect();
            // Pack two consecutive `u8` values into a single `u8`
            let _packed_b_u8: Vec<u8> = b_u8
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
            println!("Size: {}", size);

            let result_u8_chunk = dot_product_u8_chunk(&ab_u8);
            let result_u8_scalar = dot_product_u8_zipped(&ab_u8);
            println!("u8 results:");
            println!("  Chunk:      {}", result_u8_chunk);
            println!("  Scalar:     {}", result_u8_scalar);
            #[cfg(target_arch = "x86_64")]
            {
                let result_avx2 = unsafe { x86_64::dot_product_u8_avx2(&a_u8, &b_u8) };
                println!("  AVX2:       {}", result_avx2);
            }
            #[cfg(target_arch = "aarch64")]
            {
                let result_avx2 = unsafe { arm64::dot_product_u8_neon(&a_u8, &b_u8) };
                println!("  NEON:       {}", result_avx2);
            }

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
