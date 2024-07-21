use std::arch::aarch64::*;

fn print_uint8x16_t(name: &str, value: uint8x16_t) {
    let mut array = [0u8; 32];
    unsafe {
        vst1q_u8(array.as_mut_ptr() as *mut u8, value);
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

#[target_feature(enable = "neon")]
pub unsafe fn dot_product_u8_neon(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    const CHUNK_SIZE: usize = 16;

    let mut sum = vdupq_n_u16(0);
    let mut i = 0;

    while i + CHUNK_SIZE <= len {
        // Load 16 bytes from each slice into NEON registers
        let a_chunk = vld1q_u8(a.as_ptr().add(i));
        let b_chunk = vld1q_u8(b.as_ptr().add(i));

        // Multiply and accumulate lower 8 bytes
        sum = vmlal_u8(sum, vget_low_u8(a_chunk), vget_low_u8(b_chunk));
        // Multiply and accumulate higher 8 bytes
        sum = vmlal_u8(sum, vget_high_u8(a_chunk), vget_high_u8(b_chunk));
        i += CHUNK_SIZE;
    }

    // Horizontally add the elements of the sum vector
    let mut result = unsafe { vaddlvq_u32(vpaddlq_u16(sum)) };

    // Process remaining elements
    for j in i..a.len() {
        result += (a[j] as u64) * (b[j] as u64);
    }

    result
}
