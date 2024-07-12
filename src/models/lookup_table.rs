use super::rpc::VectorIdValue;
use super::types::{NodeRef, VectorId};
use crate::models::rpc::Vector;
use crate::models::types::VectorQt;
use async_std::stream::Cloned;
use dashmap::DashMap;
use futures::future::{join_all, BoxFuture, FutureExt};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
// Define the lookup table size
const U16_TABLE_SIZE: usize = u16::MAX as usize + 1;

// Create a static lookup table for u16 values
static mut U16_LOOKUP_TABLE: [u32; U16_TABLE_SIZE] = [0; U16_TABLE_SIZE];

// Function to initialize the lookup table for u16 values
pub fn initialize_u16_lookup_table() {
    for i in 0..U16_TABLE_SIZE {
        unsafe {
            U16_LOOKUP_TABLE[i] = i.count_ones();
        }
    }
}

#[inline]
pub fn shift_and_accumulate(value: u32) -> u32 {
    let high = (value >> 16) as u16;
    let low = (value & 0xFFFF) as u16;
    unsafe { U16_LOOKUP_TABLE[high as usize] + U16_LOOKUP_TABLE[low as usize] }
}
