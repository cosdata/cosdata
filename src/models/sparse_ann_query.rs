use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::models::buffered_io::BufIoError;

use crate::models::types::SparseVector;
use std::cmp::Ordering;

use super::inverted_index::InvertedIndexRoot;
use super::inverted_index_idf::{InvertedIndexIDFRoot, TermQuotient};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparseAnnResult {
    pub vector_id: u32,
    pub similarity: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparseAnnIDFResult {
    pub document_id: u32,
    pub score: f32,
}

impl Eq for SparseAnnResult {}
impl Eq for SparseAnnIDFResult {}

impl PartialOrd for SparseAnnResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SparseAnnResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other.similarity.cmp(&self.similarity)
    }
}

impl PartialOrd for SparseAnnIDFResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SparseAnnIDFResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.total_cmp(&self.score)
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
                    let pagepool = unsafe { &*node.data }
                        .try_get_data(&index.cache, node.dim_index)?
                        .map
                        .lookup(&key);
                    if let Some(pagepool) = pagepool {
                        for x in pagepool.iter() {
                            let vec_id = x;
                            let dot_product = dot_products.entry(vec_id).or_insert(0u32);
                            *dot_product += quantized_query_value * key as u32;
                        }
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

    pub fn search_bm25(
        self,
        index: &InvertedIndexIDFRoot,
        quantization_bits: u8,
        early_terminate_threshold: f32,
        reranking_factor: usize,
        k: Option<usize>,
    ) -> Result<(Vec<SparseAnnIDFResult>, Vec<f32>), BufIoError> {
        let mut results_map: FxHashMap<u32, f32> = FxHashMap::default();
        let max_key = ((1u32 << quantization_bits) - 1) as u8;
        let total_documents_count = index
            .total_documents_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let most_common_term_documents_count = index
            .most_common_term_documents_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let least_common_term_documents_count = (index
            .least_common_term_documents_count
            .load(std::sync::atomic::Ordering::Relaxed)
            >> 32) as u32;
        let max_idf = get_idf(total_documents_count, least_common_term_documents_count);
        let min_idf = get_idf(total_documents_count, most_common_term_documents_count);
        let absolute_max_idf = get_idf(total_documents_count, 0);
        let mut idf_list = Vec::with_capacity(self.query_vector.entries.len());

        for (term_hash, _query_tf) in self.query_vector.entries {
            // Split the hash dimension
            let storage_dim = term_hash & (u16::MAX as u32);
            let quotient = (term_hash >> 16) as TermQuotient;

            // Find node for this storage dimension
            let idf = if let Some(node) = index.find_node(storage_dim) {
                // Get node data
                let node_data =
                    unsafe { &*node.data }.try_get_data(&index.cache, node.dim_index)?;

                // Process documents containing this term
                if let Some(inner_map) = node_data.map.lookup(&quotient) {
                    let documents_containing_term = inner_map
                        .documents_count
                        .load(std::sync::atomic::Ordering::Relaxed);
                    // Get IDF for this term
                    let idf = get_idf(total_documents_count, documents_containing_term);
                    let idf_fraction = ((idf - min_idf) / (max_idf - min_idf)).clamp(0.0, 1.0);
                    // println!("idf_fraction: {idf_fraction}, idf: {idf}, doc_count: {documents_containing_term}, min_idf: {min_idf}, max_doc_count: {most_common_term_documents_count}, max_idf: {max_idf}, min_doc_count: {least_common_term_documents_count}");
                    let early_terminate_value =
                        (early_terminate_threshold * (1.0 - idf_fraction) * max_key as f32) as u8;
                    // println!("early_terminate_value: {}", early_terminate_value);
                    for quantized_value in early_terminate_value..=max_key {
                        if let Some(vector_ids) = inner_map.frequency_map.lookup(&quantized_value) {
                            // For each document containing this term
                            for doc_id in vector_ids.iter() {
                                // Calculate BM25 term weight
                                let term_freq = ((quantized_value as f32)
                                    * (1.2 / (1u32 << quantization_bits) as f32))
                                    + 0.8;
                                let bm25_weight = idf * term_freq;

                                // Accumulate score
                                *results_map.entry(doc_id).or_insert(0.0) += bm25_weight;
                            }
                        }
                    }
                    idf
                } else {
                    absolute_max_idf
                }
            } else {
                absolute_max_idf
            };

            idf_list.push(idf);
        }

        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<SparseAnnIDFResult> = results_map
            .into_iter()
            .map(|(vector_id, score)| SparseAnnIDFResult {
                document_id: vector_id,
                score,
            })
            .collect();
        if let Some(k) = k {
            let k_with_reranking = k * reranking_factor;
            if results.len() > k_with_reranking {
                // Use partial_sort for top K, faster than full sort
                results
                    .select_nth_unstable_by(k_with_reranking, |a, b| b.score.total_cmp(&a.score));
                results.truncate(k_with_reranking);
            }
        }
        results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        Ok((results, idf_list))
    }
}

fn get_idf(documents_count: u32, documents_containing_term: u32) -> f32 {
    (((documents_count - documents_containing_term) as f32 + 0.5)
        / (documents_containing_term as f32 + 0.5))
        .ln_1p()
}
