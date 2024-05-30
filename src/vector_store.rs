use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type NumericValue = Vec<Vec<i32>>; // Two-dimensional vector

type VectorHash = String; // Assuming VectorHash is a String, replace with appropriate type if different

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VectorTreeNode {
    vector_list: NumericValue, // Two-dimensional vector
    neighbors: Vec<(VectorHash, f64)>, // neighbor, cosine distance
}

#[derive(Debug, Clone)]
struct VectorStore {
    database_name: String,
    root_vec: (VectorHash, NumericValue), // Two-dimensional vector
    cache: HashMap<(i8, VectorHash), (Option<VectorTreeNode>, Arc<Mutex<()>>)>, // (level, vector), map prefixnorm hash
    max_cache_level: i8,
}

#[derive(Debug, Clone)]
struct VectorEmbedding {
    raw_vec: NumericValue, // Two-dimensional vector
    hash_vec: VectorHash,
}

fn cosine_coalesce(x: &[i32], y: &[i32]) -> f64 {
    let mut dp = 0;
    let mut pma = 0;
    let mut pmb = 0;

    for (&a, &b) in x.iter().zip(y.iter()) {
        let (dp_i, pma_i, pmb_i) = compute_cosine_similarity(a, b, 16);
        dp += dp_i;
        pma += pma_i;
        pmb += pmb_i;
    }

    f64::from(dp) / (f64::sqrt(f64::from(pma)) * f64::sqrt(f64::from(pmb)))
}


fn sum_components(tuple_list: &[(i32, i32, i32)]) -> (i32, i32, i32) {
    let (mut sum_x, mut sum_y, mut sum_z) = (0, 0, 0);

    for &(x, y, z) in tuple_list {
        sum_x += x;
        sum_y += y;
        sum_z += z;
    }

    (sum_x, sum_y, sum_z)
}


fn to_float_flag(x: f32) -> i32 {
    if x >= 0.0 {
        1
    } else {
        0
    }
}

fn quantize(fins: &[f32]) -> Vec<Vec<i32>> {
    let mut quantized = Vec::with_capacity((fins.len() + 15) / 16);
    let mut chunk = Vec::with_capacity(16);

    for &f in fins {
        chunk.push(to_float_flag(f));
        if chunk.len() == 16 {
            quantized.push(chunk.clone());
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        quantized.push(chunk);
    }

    quantized
}
