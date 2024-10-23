use super::buffered_io::BufIoError;
use super::dot_product::x86_64::dot_product_u8_avx2;
use super::lazy_load::LazyItem;
use super::rpc::VectorIdValue;
use super::types::{MergedNode, MetricResult, VectorId};
use crate::distance::DistanceError;
use crate::models::rpc::Vector;
use crate::models::types::PropState;
use crate::models::types::VectorQt;
use crate::quantization::QuantizationError;
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub fn dot_product_u8_avx2_fma(a: &[u8], b: &[u8]) -> u64 {
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

pub struct CosResult {
    pub dotprod: i32,
    pub premag_a: i32,
    pub premag_b: i32,
}
pub fn dot_product_u8_xxx(src: &[(u8, u8)], dst: &mut [u64]) {
    let dst_known_bounds = &mut dst[0..src.len()];
    let size = 8;
    let len = src.len();
    // Process chunks of 8
    let mut i = 0;
    while i + size <= len {
        dst_known_bounds[i] = ((src[0].0) * (src[0].1)) as u64;
        dst_known_bounds[i + 1] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 2] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 3] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 4] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 5] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 6] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        dst_known_bounds[i + 7] = ((src[i + 1].0) * (src[i + 1].1)) as u64;
        i += size;
    }
    // Handle remaining elements
    while i < len {
        dst_known_bounds[i] = (src[i].0 as u64) * (src[i].1 as u64);
        i += 1;
    }
}
pub fn dot_product_f32_xxx(src: &[(f32, f32)], dst: &mut [f32]) {
    let dst_known_bounds = &mut dst[0..src.len()];
    let size = 4;
    let len = src.len();
    // Process chunks of 8
    let mut i = 0;
    while i + size <= len {
        dst_known_bounds[i] = (src[0].0) * (src[0].1);
        dst_known_bounds[i + 1] = (src[i + 1].0) * (src[i + 1].1);
        dst_known_bounds[i + 2] = (src[i + 1].0) * (src[i + 1].1);
        dst_known_bounds[i + 3] = (src[i + 1].0) * (src[i + 1].1);
        i += size;
    }
    // Handle remaining elements
    while i < len {
        dst_known_bounds[i] = (src[i].0) * (src[i].1);
        i += 1;
    }
}

pub fn dot_product_f32_chunk(src: &[(f32, f32)], _dst: &mut [f32]) -> f32 {
    let mut d: f32 = 0.0;
    let size = 4;

    // Process chunks of 4
    for chunk in src.chunks_exact(size) {
        let mut local_sum: f32 = 0.0;
        local_sum += chunk[0].0 * chunk[0].1;
        local_sum += chunk[1].0 * chunk[1].1;
        local_sum += chunk[2].0 * chunk[2].1;
        local_sum += chunk[3].0 * chunk[3].1;
        d += local_sum;
    }

    // Handle remaining elements
    for &(a, b) in src.chunks_exact(size).remainder() {
        d += a * b;
    }

    d
}
pub fn dot_product_u8_chunk(src: &[(u8, u8)]) -> u64 {
    let mut d: u64 = 0;
    let size = 8;

    // Process chunks of 8
    for chunk in src.chunks_exact(size) {
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
    for &(a, b) in src.chunks_exact(size).remainder() {
        d += (a as u64) * (b as u64);
    }

    d
}
pub fn dot_product_a(src: &[(f32, f32)], dst: &mut [f32]) -> f32 {
    let mut d: f32 = 0.0;
    for (_dst_sample, src_sample) in dst.iter_mut().zip(src.iter()) {
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

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(vec: &[f32]) -> f32 {
    vec.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b) / (magnitude(a) * magnitude(b))
}

pub fn get_magnitude_plus_quantized_vec111(quant_vec: Vec<Vec<u32>>, _size: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(quant_vec.len());

    for vecx in quant_vec {
        let premag: u32 = vecx.iter().fold(0, |acc, &val| acc + val.count_ones());
        result.push(premag);
    }

    result
}

pub fn get_magnitude_plus_quantized_vec(quant_vec: &[Vec<u32>], _size: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(quant_vec.len());

    for vecx in quant_vec {
        let premag: usize = vecx
            .iter()
            .fold(0, |acc, &val| acc + val.count_ones() as usize);
        result.push(premag);
    }

    result
}
pub fn cosine_similarity_scalar_u8(x: &VectorQt, y: &VectorQt) -> f32 {
    if let (
        VectorQt::UnsignedByte {
            mag: x_mag,
            quant_vec: x_vec,
        },
        VectorQt::UnsignedByte {
            mag: y_mag,
            quant_vec: y_vec,
        },
    ) = (x, y)
    {
        let dot_product = unsafe { dot_product_u8_avx2(x_vec, y_vec) };
        dot_product as f32 / ((*x_mag as f32).sqrt() * (*y_mag as f32).sqrt())
    } else {
        panic!("cosine_similarity_scalar_u8 called with non-UnsignedByte VectorQt")
    }
}
pub fn cosine_coalesce(x: &VectorQt, y: &VectorQt, length: usize) -> f32 {
    if let (
        VectorQt::SubByte {
            quant_vec: x_vec,
            resolution: x_res,
            ..
        },
        VectorQt::SubByte {
            quant_vec: y_vec,
            resolution: y_res,
            ..
        },
    ) = (x, y)
    {
        if x_res != y_res {
            panic!("Resolution mismatch in cosine_coalesce");
        }

        let parts = 2_usize.pow(*x_res as u32);
        let mut final_result: usize = 0;

        for index in 0..parts {
            let sum: usize = x_vec[index]
                .iter()
                .zip(&y_vec[index])
                .map(|(&x_item, &y_item)| (x_item ^ y_item).count_ones() as usize)
                .sum();
            final_result += sum << index;
        }

        final_result as f32 / length as f32
    } else {
        panic!("cosine_coalesce called with non-SubByte VectorQt")
    }
}
pub fn cosine_similarity_qt(
    x: &VectorQt,
    y: &VectorQt,
    length: usize,
) -> Result<f32, WaCustomError> {
    match (x, y) {
        (VectorQt::SubByte { .. }, VectorQt::SubByte { .. }) => Ok(cosine_coalesce(x, y, length)),
        (VectorQt::UnsignedByte { .. }, VectorQt::UnsignedByte { .. }) => {
            Ok(cosine_similarity_scalar_u8(x, y))
        }
        _ => Err(WaCustomError::QuantizationMismatch),
    }
}
//////
#[inline]

fn to_float_flag(x: f32, bits_per_value: usize, step: f32) -> Vec<bool> {
    let mut n = ((x + 1.0) / step).floor() as usize;
    let mut result = vec![false; bits_per_value];
    // Fill the vector with bits from the least significant to the most significant
    for i in (0..bits_per_value).rev() {
        result[i] = (n & 1) == 1;
        n >>= 1;
    }

    result
}

pub fn simp_quant(v: &[f32]) -> Result<Vec<u8>, QuantizationError> {
    let (out_of_range, has_negative) = v.iter().fold((false, false), |(oor, neg), &x| {
        (oor || x > 1.0 || x < -1.0, neg || x < 0.0)
    });
    if out_of_range {
        return Err(QuantizationError::InvalidInput(String::from(
            "Values sent in vector for simp_quant are out of range [-1,+1]",
        )));
    }
    let res: Vec<u8> = v
        .iter()
        .map(|&x| {
            let y = if has_negative { x + 1.0 } else { x };
            (y * 255.0).round() as u8
        })
        .collect();
    Ok(res)
}

pub fn mag_square_u8(vec: &[u8]) -> u32 {
    vec.iter().map(|&x| x as u32 * x as u32).sum()
}

pub fn quantize_to_u8_bits(fins: &[f32], resolution: u8) -> Vec<Vec<u8>> {
    let bits_per_value = resolution as usize;
    let parts = 2_usize.pow(bits_per_value as u32);
    let step = 2.0 / parts as f32;
    let u32s_per_value = (fins.len()) / 8;
    let mut quantized: Vec<Vec<u8>> = vec![Vec::with_capacity(u32s_per_value); bits_per_value];

    let mut current_u32s: Vec<u8> = vec![0; bits_per_value];
    let mut bit_index: usize = 0;

    for &f in fins {
        let flags = to_float_flag(f, bits_per_value, step);

        for bit_position in 0..bits_per_value {
            if flags[bit_position] {
                current_u32s[bit_position] |= 1 << bit_index;
            }
        }
        bit_index += 1;

        if bit_index == 32 {
            for bit_position in 0..bits_per_value {
                quantized[bit_position].push(current_u32s[bit_position]);
                current_u32s[bit_position] = 0;
            }
            bit_index = 0;
        }
    }

    if bit_index > 0 {
        for bit_position in 0..bits_per_value {
            quantized[bit_position].push(current_u32s[bit_position]);
        }
    }

    quantized
}

#[derive(Debug, Clone)]
pub enum WaCustomError {
    DatabaseError(String),
    SerializationError(String),
    UpsertFailed,
    InvalidParams,
    NodeError(String),
    NeighborError(String, Option<VectorId>),
    LockError(String),
    QuantizationMismatch,
    LazyLoadingError(String),
    TrainingFailed,
    CalculationError,
    FsError(String),
    DeserializationError(String),
    // put it in `Arc` to make it cloneable
    BufIo(Arc<BufIoError>),
}

impl fmt::Display for WaCustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaCustomError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            WaCustomError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            WaCustomError::UpsertFailed => write!(f, "Failed to upsert vectors"),
            WaCustomError::InvalidParams => write!(f, "Invalid params in request"),
            WaCustomError::NodeError(msg) => write!(f, "Node error: {}", msg),
            WaCustomError::NeighborError(msg, id) => match id {
                Some(vec_id) => write!(f, "Neighbor error: {} for ID {}", msg, vec_id),
                None => write!(f, "Neighbor error: {}", msg),
            },
            WaCustomError::LockError(msg) => write!(f, "Lock error: {}", msg),
            WaCustomError::QuantizationMismatch => write!(f, "Quantization mismatch"),
            WaCustomError::LazyLoadingError(msg) => write!(f, "Lazy loading error: {}", msg),
            WaCustomError::TrainingFailed => write!(f, "Training failed"),
            WaCustomError::CalculationError => write!(f, "Calculation error"),
            WaCustomError::FsError(err) => write!(f, "FS error: {}", err),
            WaCustomError::DeserializationError(err) => write!(f, "Deserialization error: {}", err),
            WaCustomError::BufIo(err) => write!(f, "Buffer IO error: {}", err),
        }
    }
}

impl From<QuantizationError> for WaCustomError {
    fn from(value: QuantizationError) -> Self {
        match value {
            QuantizationError::InvalidInput(_) => WaCustomError::InvalidParams,
            QuantizationError::TrainingFailed => WaCustomError::TrainingFailed,
        }
    }
}

impl From<DistanceError> for WaCustomError {
    fn from(value: DistanceError) -> Self {
        match value {
            DistanceError::StorageMismatch => WaCustomError::QuantizationMismatch,
            DistanceError::CalculationError => WaCustomError::CalculationError,
        }
    }
}

impl From<BufIoError> for WaCustomError {
    fn from(error: BufIoError) -> Self {
        Self::BufIo(Arc::new(error))
    }
}

pub fn hash_float_vec(vec: Vec<f32>) -> Vec<u8> {
    // Create a new hasher instance
    let mut hasher = Sha256::new();

    // Convert the Vec<f32> to a byte representation
    for &num in &vec {
        // Convert each f32 to its byte representation and update the hasher
        hasher.update(&num.to_le_bytes());
    }

    // Finalize the hash and return the result as a Vec<u8>
    hasher.finalize().to_vec()
}

pub fn get_max_insert_level(x: f64, levels: Arc<Vec<(f64, i32)>>) -> i32 {
    let lst = levels.iter();
    match lst.clone().find(|(value, _)| x >= *value) {
        Some((_, index)) => *index,
        None => panic!("No matching element found"),
    }
}

pub fn add_option_vecs(
    a: &Option<Vec<(LazyItem<MergedNode>, MetricResult)>>,
    b: &Option<Vec<(LazyItem<MergedNode>, MetricResult)>>,
) -> Option<Vec<(LazyItem<MergedNode>, MetricResult)>> {
    match (a, b) {
        (None, None) => None,
        (Some(vec), None) | (None, Some(vec)) => Some(vec.clone()),
        (Some(vec1), Some(vec2)) => {
            let mut combined = vec1.clone();
            combined.extend(vec2.iter().cloned());
            Some(combined)
        }
    }
}

// Function to convert VectorIdValue to VectorId
pub fn convert_value(id_value: VectorIdValue) -> VectorId {
    match id_value {
        VectorIdValue::StringValue(s) => VectorId::Str(s),
        VectorIdValue::IntValue(i) => VectorId::Int(i),
    }
}

// Function to convert VectorId to VectorIdValue
fn convert_id(id: VectorId) -> VectorIdValue {
    match id {
        VectorId::Str(s) => VectorIdValue::StringValue(s),
        VectorId::Int(i) => VectorIdValue::IntValue(i),
    }
}

// Function to convert the Option<Vec<(VectorId, _)>> to Option<Vec<(VectorIdValue, _)>>
pub fn convert_option_vec(
    input: Option<Vec<(VectorId, MetricResult)>>,
) -> Option<Vec<(VectorIdValue, MetricResult)>> {
    input.map(|vec| {
        vec.into_iter()
            .map(|(id, value)| (convert_id(id), value))
            .collect()
    })
}

// Function to convert Vec<Vector> to Vec<(VectorIdValue, Vec<f32>)>
pub fn convert_vectors(vectors: Vec<Vector>) -> Vec<(VectorIdValue, Vec<f32>)> {
    vectors
        .into_iter()
        .map(|vector| (vector.id.clone(), vector.values))
        .collect()
}

pub fn remove_duplicates_and_filter(
    input: Option<Vec<(LazyItem<MergedNode>, MetricResult)>>,
) -> Option<Vec<(VectorId, MetricResult)>> {
    input.map(|vec| {
        let mut seen = HashSet::new();
        vec.into_iter()
            .filter_map(|(lazy_item, similarity)| {
                if let LazyItem::Valid { mut data, .. } = lazy_item {
                    if let Some(data) = data.get() {
                        let mut prop_arc = data.prop.clone();
                        if let PropState::Ready(node_prop) = prop_arc.get() {
                            let id = &node_prop.id;
                            if let VectorId::Int(s) = id {
                                if *s == -1 {
                                    return None;
                                }
                            }
                            if seen.insert(id.clone()) {
                                Some((id.clone(), similarity))
                            } else {
                                None
                            }
                        } else {
                            None // PropState is Pending
                        }
                    } else {
                        None // data is None
                    }
                } else {
                    None // LazyItem is Invalid
                }
            })
            .collect()
    })
}

pub fn generate_tuples(x: f64, max_cache_level: u8) -> Vec<(f64, i32)> {
    let mut result = Vec::new();
    for n in (0..=max_cache_level).rev() {
        let first_item = 1.0 - x.powi(-(n as i32));
        let second_item = n as i32;
        result.push((first_item, second_item));
    }
    result
}

pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    t.hash(&mut hasher);
    hasher.finish()
}

// Extract VectorId values for hashing purposes
pub fn extract_ids(neighbors: &[(VectorId, f32)]) -> Vec<VectorId> {
    neighbors.iter().map(|(id, _)| id.clone()).collect()
}

// Optional: Implement From trait for more idiomatic conversion

impl From<VectorId> for VectorIdValue {
    fn from(vector_id: VectorId) -> Self {
        match vector_id {
            VectorId::Str(s) => VectorIdValue::StringValue(s),
            VectorId::Int(i) => VectorIdValue::IntValue(i),
        }
    }
}

impl From<VectorIdValue> for VectorId {
    fn from(vector_id_value: VectorIdValue) -> Self {
        match vector_id_value {
            VectorIdValue::StringValue(s) => VectorId::Str(s),
            VectorIdValue::IntValue(i) => VectorId::Int(i),
        }
    }
}

pub fn cat_maybes<T>(iter: impl Iterator<Item = Option<T>>) -> Vec<T> {
    iter.flat_map(|maybe| maybe).collect()
}

pub fn tapered_total_hops(hops: u8, cur_level: u8, max_level: u8) -> u8 {
    //div by 2
    if cur_level > max_level >> 1 {
        return hops;
    } else {
        // div by 4
        if cur_level > max_level >> 2 {
            return 3 * (hops >> 2); // 3/4
        } else {
            return hops >> 1; // 1/2
        }
    }
}
//typically skips is 1 while near
pub fn tapered_skips(skips: i8, cur_distance: i8, max_distance: i8) -> i8 {
    // Calculate the distance ratio (0.0 to 1.0)
    let distance_ratio = cur_distance as f32 / max_distance as f32;

    // Use match expression for efficient logic based on distance ratio
    match distance_ratio {
        ratio if ratio < 0.25 => skips,
        ratio if ratio < 0.5 => skips * 2,
        ratio if ratio < 0.75 => skips * 3,
        _ => skips * 4, // Distance ratio >= 0.75
    }
}

pub fn tuple_to_string(tuple: (u32, u32)) -> String {
    format!("{}_{}", tuple.0, tuple.1)
}
