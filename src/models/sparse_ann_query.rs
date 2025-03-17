use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::models::buffered_io::BufIoError;

use crate::models::types::SparseVector;
use std::cmp::Ordering;

use super::inverted_index::InvertedIndexRoot;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparseAnnResult {
    pub vector_id: u32,
    pub similarity: u32,
}

impl Eq for SparseAnnResult {}

impl PartialOrd for SparseAnnResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(other.similarity.cmp(&self.similarity))
    }
}

impl Ord for SparseAnnResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other.similarity.cmp(&self.similarity)
    }
}

#[derive(Clone)]
pub struct SparseAnnQueryBasic {
    /// Query vector is a pair of non-zero values and its dimension
    query_vector: SparseVector,
}

impl SparseAnnQueryBasic {
    pub fn new(query_vector: SparseVector) -> Self {
        SparseAnnQueryBasic { query_vector }
    }

    pub fn sequential_search(
        self,
        index: &InvertedIndexRoot,
        // 4, 5, 6
        quantization_bits: u8,
        values_upper_bound: f32,
        early_terminate_threshold: f32,
        reranking_factor: usize,
        k: Option<usize>,
    ) -> Result<Vec<SparseAnnResult>, BufIoError> {
        let mut dot_products = FxHashMap::default();
        // same as `1` quantized
        let one_quantized = ((1u32 << quantization_bits) - 1) as u8;
        let early_terminate_value = ((1u32 << quantization_bits) as f32 * early_terminate_threshold)
            .min(u8::MAX as f32) as u8;
        let low_threshold = (early_terminate_threshold * (1u32 << quantization_bits) as f32) as u32;

        let mut sorted_query_dims = self.query_vector.entries;
        sorted_query_dims.sort_by(|(_, a), (_, b)| b.total_cmp(a));

        // Iterate over the query vector dimensions
        for &(dim_index, dim_value) in &sorted_query_dims {
            let Some(node) = index.find_node(dim_index) else {
                continue;
            };
            let quantized_query_value = node.quantize(dim_value, values_upper_bound) as u32;

            if quantized_query_value > low_threshold {
                // High quantized value
                // Iterate through the full list of values for this dimension
                for key in (0..=one_quantized).rev() {
                    let mut current_versioned_pagepool = unsafe { &*node.data }
                        .try_get_data(&index.cache, node.dim_index)?
                        .map
                        .lookup(&key);
                    while let Some(versioned_pagepool) = current_versioned_pagepool {
                        for x in versioned_pagepool.pagepool.inner.read().unwrap().iter() {
                            for x in x.iter() {
                                let vec_id = *x;
                                let dot_product = dot_products.entry(vec_id).or_insert(0u32);
                                *dot_product += quantized_query_value * key as u32;
                            }
                        }
                        current_versioned_pagepool =
                            versioned_pagepool.next.read().unwrap().clone();
                    }
                }
            } else {
                // Low quantized value
                // Iterate through the map/list ONLY for until a certain threshold (say 3/4th)
                // of quantized keys (i.e. 48..64 for 6 bit quantization).

                for key in (early_terminate_value..=one_quantized).rev() {
                    let mut current_versioned_pagepool = unsafe { &*node.data }
                        .try_get_data(&index.cache, node.dim_index)?
                        .map
                        .lookup(&key);
                    while let Some(versioned_pagepool) = current_versioned_pagepool {
                        for x in versioned_pagepool.pagepool.inner.read().unwrap().iter() {
                            for x in x.iter() {
                                let vec_id = *x;

                                let dot_product = dot_products.entry(vec_id).or_insert(0u32);
                                *dot_product += quantized_query_value * key as u32;
                            }
                        }
                        current_versioned_pagepool =
                            versioned_pagepool.next.read().unwrap().clone();
                    }
                }
            }
        }

        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<SparseAnnResult> = dot_products
            .into_iter()
            .map(|(vector_id, similarity)| SparseAnnResult {
                vector_id,
                similarity,
            })
            .collect();
        if let Some(k) = k {
            let k_with_reranking = k * reranking_factor;
            if results.len() > k_with_reranking {
                // Use partial_sort for top K, faster than full sort
                results.select_nth_unstable_by(k_with_reranking, |a, b| {
                    b.similarity.cmp(&a.similarity)
                });
                results.truncate(k_with_reranking);
            }
        }
        Ok(results)
    }
}
