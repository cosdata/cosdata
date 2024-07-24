use std::hash::{Hash, Hasher};
// Define the lookup table size
const U16_TABLE_SIZE: usize = u16::MAX as usize + 1;

// Create a static lookup table for u16 values
static mut U16_LOOKUP_TABLE: [u32; U16_TABLE_SIZE] = [0; U16_TABLE_SIZE];

// Function to initialize the lookup table for u16 values
pub fn initialize_u16_lookup_table() {
    for i in 0..U16_TABLE_SIZE {
        unsafe {
            U16_LOOKUP_TABLE[i] = shift_and_accumulate_u16(i as u16);
        }
    }
}

// Helper function to compute shift_and_accumulate for u16 values
fn shift_and_accumulate_u16(value: u16) -> u32 {
    let mut result: u32 = 0;
    result += x_function(15 & (value as u32 >> 0));
    result += x_function(15 & (value as u32 >> 4));
    result += x_function(15 & (value as u32 >> 8));
    result += x_function(15 & (value as u32 >> 12));
    result
}

// x_function remains the same
pub fn x_function(value: u32) -> u32 {
    match value {
        0 => 0,
        1 => 1,
        2 => 1,
        3 => 2,
        4 => 1,
        5 => 2,
        6 => 2,
        7 => 3,
        8 => 1,
        9 => 2,
        10 => 2,
        11 => 3,
        12 => 2,
        13 => 3,
        14 => 3,
        15 => 4,
        _ => 0, // Invalid input
    }
}

#[inline]
pub fn shift_and_accumulate(value: u32) -> u32 {
    let high = (value >> 16) as u16;
    let low = (value & 0xFFFF) as u16;
    unsafe { U16_LOOKUP_TABLE[high as usize] + U16_LOOKUP_TABLE[low as usize] }
}
