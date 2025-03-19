use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub unsafe fn dot_product_u8_neon(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    const CHUNK_SIZE: usize = 16;

    let mut sum1 = vdupq_n_u32(0);
    let mut sum2 = vdupq_n_u32(0);
    let mut i = 0;

    while i + CHUNK_SIZE <= len {
        let a_chunk = vld1q_u8(a.as_ptr().add(i));
        let b_chunk = vld1q_u8(b.as_ptr().add(i));

        // Split into low and high parts
        let a_low = vget_low_u8(a_chunk);
        let a_high = vget_high_u8(a_chunk);
        let b_low = vget_low_u8(b_chunk);
        let b_high = vget_high_u8(b_chunk);

        // Zero-extend to u16x8
        let a_low_u16 = vmovl_u8(a_low);
        let a_high_u16 = vmovl_u8(a_high);
        let b_low_u16 = vmovl_u8(b_low);
        let b_high_u16 = vmovl_u8(b_high);

        // Multiply to get products
        let product_low = vmulq_u16(a_low_u16, b_low_u16);
        let product_high = vmulq_u16(a_high_u16, b_high_u16);

        // Accumulate pairwise sums into 32-bit accumulators
        sum1 = vpadalq_u16(sum1, product_low);
        sum2 = vpadalq_u16(sum2, product_high);

        i += CHUNK_SIZE;
    }

    // Combine the two accumulators
    let sum_total = vaddq_u32(sum1, sum2);

    // Horizontally add the elements of the sum vector
    let mut result = vaddvq_u32(sum_total) as u64;

    // Process remaining elements
    for j in i..len {
        result += (a[j] as u64) * (b[j] as u64);
    }

    result
}
