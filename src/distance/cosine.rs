use serde::{Deserialize, Serialize};

use super::{DistanceError, DistanceFunction};
use crate::{
    models::{
        dot_product::{
            dot_product_binary, dot_product_f16, dot_product_f32, dot_product_octal,
            dot_product_quaternary, dot_product_u8,
        },
        types::{Metadata, ReplicaNodeKind, VectorData},
    },
    storage::Storage,
};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize, PartialOrd)]
pub struct CosineDistance(pub f32);

impl DistanceFunction for CosineDistance {
    type Item = Self;
    fn calculate(
        &self,
        _x: &VectorData,
        _y: &VectorData,
        _is_indexing: bool,
    ) -> Result<Self::Item, DistanceError> {
        // TODO: Implement cosine distance
        unimplemented!("Cosine distance is not implemented yet");
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize, PartialOrd)]
pub struct CosineSimilarity(pub f32);

impl DistanceFunction for CosineSimilarity {
    type Item = Self;
    fn calculate(
        &self,
        x: &VectorData,
        y: &VectorData,
        // @TODO(vineet): Check if is_index can be removed now
        _is_indexing: bool,
    ) -> Result<Self::Item, DistanceError> {
        let x_kind = x.replica_node_kind();
        let y_kind = y.replica_node_kind();
        let sim = match (y_kind, x_kind) {
            (ReplicaNodeKind::Pseudo, ReplicaNodeKind::Pseudo) => {
                let x_metadata = x.metadata.unwrap();
                let y_metadata = y.metadata.unwrap();
                if x_metadata.mbits == y_metadata.mbits {
                    CosineSimilarity(1.0)
                } else {
                    CosineSimilarity(0.0)
                }
            }
            (ReplicaNodeKind::Pseudo, ReplicaNodeKind::Base) => {
                // A base node should never match an existing pseudo
                // node
                CosineSimilarity(0.0)
            }
            (ReplicaNodeKind::Pseudo, ReplicaNodeKind::Metadata) => {
                let x_metadata = x.metadata.unwrap();
                let y_metadata = y.metadata.unwrap();
                if x_metadata.mbits == y_metadata.mbits {
                    CosineSimilarity(1.0)
                } else {
                    CosineSimilarity(0.0)
                }
            }
            (ReplicaNodeKind::Base, ReplicaNodeKind::Pseudo) => {
                if y.is_pseudo_root() {
                    CosineSimilarity(1.0)
                } else {
                    CosineSimilarity(-1.0)
                }
            }
            (ReplicaNodeKind::Base, ReplicaNodeKind::Base) => {
                cosine_similarity(x.quantized_vec, y.quantized_vec, None)?
            }
            (ReplicaNodeKind::Metadata, ReplicaNodeKind::Metadata) => {
                let x_metadata = x.metadata.unwrap();
                let y_metadata = y.metadata.unwrap();
                if x_metadata.mbits == y_metadata.mbits {
                    cosine_similarity(x.quantized_vec, y.quantized_vec, None)?
                } else {
                    CosineSimilarity(0.0)
                }
            }
            (ReplicaNodeKind::Base, ReplicaNodeKind::Metadata) => CosineSimilarity(0.0),
            (ReplicaNodeKind::Metadata, ReplicaNodeKind::Pseudo) => {
                // This case is not possible
                unreachable!()
            }
            (ReplicaNodeKind::Metadata, ReplicaNodeKind::Base) => {
                // @TODO(vineet): This case is actually shouldn't be
                // possible. Check why it's happening.
                CosineSimilarity(0.0)
            }
        };
        Ok(sim)
    }
}

// @NOTE: Following function is not in use. It's kept for later
// reference if required.
#[allow(unused)]
fn metadata_dims_dot_product(
    x_metadata: &Metadata,
    y_metadata: &Metadata,
) -> Option<(f32, f32, f32)> {
    if x_metadata.mag == 0.0 || y_metadata.mag == 0.0 {
        None
    } else {
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

fn cosine_similarity(
    x_quantized: &Storage,
    y_quantized: &Storage,
    m_dot_product: Option<(f32, f32, f32)>,
) -> Result<CosineSimilarity, DistanceError> {
    match (x_quantized, y_quantized) {
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
            match &m_dot_product {
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
            match &m_dot_product {
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
            match &m_dot_product {
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
            match &m_dot_product {
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
