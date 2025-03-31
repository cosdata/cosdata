use serde::{Deserialize, Serialize};

use super::{DistanceError, DistanceFunction};
use crate::{
    models::{
        dot_product::{
            dot_product_binary, dot_product_f16, dot_product_f32, dot_product_octal,
            dot_product_quaternary, dot_product_u8,
        },
        types::VectorData,
    },
    storage::Storage,
};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize, PartialOrd)]
pub struct CosineDistance(pub f32);

impl DistanceFunction for CosineDistance {
    type Item = Self;
    fn calculate(&self, _x: &VectorData, _y: &VectorData, _is_indexing: bool) -> Result<Self::Item, DistanceError> {
        // TODO: Implement cosine distance
        unimplemented!("Cosine distance is not implemented yet");
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize, PartialOrd)]
pub struct CosineSimilarity(pub f32);

impl DistanceFunction for CosineSimilarity {
    type Item = Self;
    fn calculate(&self, x: &VectorData, y: &VectorData, is_indexing: bool) -> Result<Self::Item, DistanceError> {
        // Here we're adding metadata fields for both vectors into a
        // single tuple inside an Option that serves two purposes -
        // 1. makes it easy to test if both vectors contain metadata
        //    multiple times
        // 2. allows dot product to be computed only once
        let metadata = match (x.metadata, y.metadata) {
            (Some(x_metadata), Some(y_metadata)) => {
                // @TODO(vineet): Check if the second case (x) is
                // required. Can we assume that x is always either the
                // query vector (in case of is_indexing = false) and
                // vector to be inserted (in case of is_indexing =
                // true)?
                if y_metadata.mag == 0.0 || x_metadata.mag == 0.0 {
                    if is_indexing {
                        // When indexing, if the vector to be inserted
                        // has metadata fields but other node
                        // (existing) it's compared with is a base
                        // replica (i.e. having base dimensions
                        // resulting in mag = 0), then we fallback to
                        // comparing the combined vector + metadata
                        // dimensions.
                        None
                    } else {
                        // When querying, if the query vector has
                        // metadata fields but the existing node it's
                        // compared against is a base replica, then we
                        // directly know it's a mismatch. Hence return
                        // early.
                        return Ok(CosineSimilarity(0.0));
                    }
                } else {
                    // @NOTE: Here we are casting i32 to f32, which means
                    // truncation is possible, but it's not a concern in
                    // this case because metadata dims will either be 0,
                    // -1, 1 (query filter encoding) or high weight values
                    // (we need to make sure it's high enough to be
                    // effective but wouldn't result in truncation if
                    // casted to f32)
                    let x_mdims = x_metadata
                        .mbits
                        .iter()
                        .map(|i| *i as f32)
                        .collect::<Vec<f32>>();
                    let y_mdims = y_metadata
                        .mbits
                        .iter()
                        .map(|i| *i as f32)
                        .collect::<Vec<f32>>();
                    let m_dot_product: f32 = dot_product_f32(&x_mdims, &y_mdims);
                    Some((x_metadata.mag, y_metadata.mag, m_dot_product))
                }
            }
            _ => None,
        };

        // If not indexing (which means we're querying), we match
        // metadata dimensions on priority and short circuit if
        // there's no match
        if !is_indexing {
            // If metadata exists in both vectors, we first compute cosine
            // similarity for metadata dimensions. Only if the metadata
            // similarity is ~1 (consider a small epsilon for floating
            // point rounding), compute full similarity.
            match &metadata {
                Some((x_mag, y_mag, m_dot_product)) => {
                    let m_cos_sim = cosine_similarity_from_dot_product(*m_dot_product, *x_mag, *y_mag)?;
                    let threshold: f32 = 0.99;
                    if m_cos_sim.0 < threshold {
                        // Not close enough to 1, so return
                        // CosineSimilarity of 0 as we don't want the
                        // vectors to match
                        return Ok(CosineSimilarity(0.0));
                    }
                }
                _ => { }
            }
        }

        // Only if metadata vectors are close enough, we compute total
        // cosine similarity
        match (x.quantized_vec, y.quantized_vec) {
            (
                Storage::UnsignedByte {
                    mag: x_mag,
                    quant_vec: x_vec,
                },
                Storage::UnsignedByte {
                    mag: y_mag,
                    quant_vec: y_vec,
                },
            ) => {
                let dot_product = dot_product_u8(x_vec, y_vec) as f32;
                match &metadata {
                    Some((x_mmag, y_mmag, m_dot_product)) => cosine_similarity_with_metadata(
                        dot_product,
                        *m_dot_product,
                        *x_mag,
                        *x_mmag,
                        *y_mag,
                        *y_mmag,
                    ),
                    _ => cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag),
                }
            }
            (
                Storage::SubByte {
                    mag: x_mag,
                    quant_vec: x_vec,
                    resolution: x_res,
                },
                Storage::SubByte {
                    mag: y_mag,
                    quant_vec: y_vec,
                    resolution: y_res,
                },
            ) => {
                if x_res != y_res {
                    return Err(DistanceError::StorageMismatch);
                }
                let dot_product = match *x_res {
                    1 => dot_product_binary(x_vec, y_vec, *x_res),
                    2 => dot_product_quaternary(x_vec, y_vec, *x_res),
                    3 => dot_product_octal(x_vec, y_vec, *x_res),
                    _ => {
                        return Err(DistanceError::CalculationError);
                    }
                };
                match &metadata {
                    Some((x_mmag, y_mmag, m_dot_product)) => cosine_similarity_with_metadata(
                        dot_product,
                        *m_dot_product,
                        *x_mag,
                        *x_mmag,
                        *y_mag,
                        *y_mmag,
                    ),
                    _ => cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag),
                }
            }
            (
                Storage::HalfPrecisionFP {
                    mag: x_mag,
                    quant_vec: x_vec,
                },
                Storage::HalfPrecisionFP {
                    mag: y_mag,
                    quant_vec: y_vec,
                },
            ) => {
                let dot_product = dot_product_f16(x_vec, y_vec);
                match &metadata {
                    Some((x_mmag, y_mmag, m_dot_product)) => cosine_similarity_with_metadata(
                        dot_product,
                        *m_dot_product,
                        *x_mag,
                        *x_mmag,
                        *y_mag,
                        *y_mmag,
                    ),
                    _ => cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag),
                }
            }
            (
                Storage::FullPrecisionFP {
                    mag: x_mag,
                    vec: x_vec,
                },
                Storage::FullPrecisionFP {
                    mag: y_mag,
                    vec: y_vec,
                },
            ) => {
                let dot_product = dot_product_f32(x_vec, y_vec);
                match &metadata {
                    Some((x_mmag, y_mmag, m_dot_product)) => cosine_similarity_with_metadata(
                        dot_product,
                        *m_dot_product,
                        *x_mag,
                        *x_mmag,
                        *y_mag,
                        *y_mmag,
                    ),
                    _ => cosine_similarity_from_dot_product(dot_product, *x_mag, *y_mag),
                }
            }
            _ => Err(DistanceError::StorageMismatch),
        }
    }
}

fn cosine_similarity_from_dot_product(
    dot_product: f32,
    x_mag: f32,
    y_mag: f32,
) -> Result<CosineSimilarity, DistanceError> {
    let denominator = x_mag * y_mag;

    if denominator == 0.0 {
        Err(DistanceError::CalculationError)
    } else {
        Ok(CosineSimilarity(dot_product / denominator))
    }
}

/// Calculates cosine similarity when vector data contains metadata
fn cosine_similarity_with_metadata(
    dot_product_vec: f32,
    dot_product_m: f32,
    x_mag_vec: f32,
    x_mag_m: f32,
    y_mag_vec: f32,
    y_mag_m: f32,
) -> Result<CosineSimilarity, DistanceError> {
    // @NOTE: Since norm/mag values for metadata can be large due to
    // high weight values, we need to take care of overflow during
    // intermediate addition
    let norm_x = (x_mag_vec.powi(2) + x_mag_m.powi(2)).min(f32::MAX).sqrt();
    let norm_y = (y_mag_vec.powi(2) + y_mag_m.powi(2)).min(f32::MAX).sqrt();
    let denominator = (norm_x * norm_y).min(f32::MAX);
    if denominator == 0.0 {
        Err(DistanceError::CalculationError)
    } else {
        let numerator = dot_product_vec + dot_product_m;
        Ok(CosineSimilarity(numerator / denominator))
    }
}
