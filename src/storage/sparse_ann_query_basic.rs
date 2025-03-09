use serde::Serialize;

use super::inverted_index_sparse_ann_basic::{
    InvertedIndexSparseAnnBasic, InvertedIndexSparseAnnBasicDashMap,
    InvertedIndexSparseAnnBasicTSHashmap, InvertedIndexSparseAnnNodeBasic,
    InvertedIndexSparseAnnNodeBasicDashMap,
};
use crate::models::buffered_io::BufIoError;

use crate::models::types::{SparseQueryVector, SparseQueryVectorDimensionType, SparseVector};
use std::collections::{HashMap, HashSet};
use std::{cmp::Ordering, collections::BinaryHeap};

const K: usize = 100;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparseAnnResult {
    pub vector_id: u32,
    pub similarity: u32,
}

impl Eq for SparseAnnResult {}

impl PartialOrd for SparseAnnResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.similarity.partial_cmp(&self.similarity)
    }
}

impl Ord for SparseAnnResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct SparseAnnQueryBasic {
    /// Query vector is a pair of non-zero values and its dimension
    query_vector: SparseVector,
}

impl SparseAnnQueryBasic {
    pub fn new(query_vector: SparseVector) -> Self {
        SparseAnnQueryBasic { query_vector }
    }

    // Creates a sparse query vector with dimension types based on the posting list length.
    // This is used to optimize sparse ANN search using a cuckoo filter and two-stage search.
    fn create_sparse_query_vector(
        &self,
        index: &InvertedIndexSparseAnnBasicTSHashmap,
    ) -> Result<SparseQueryVector, BufIoError> {
        let mut posting_list_lengths: Vec<(u32, usize)> = Vec::new();
        for (dim_index, _) in &self.query_vector.entries {
            if let Some(node) = index.find_node(*dim_index) {
                let mut length = 0;
                for key in 0..64 {
                    length += unsafe { &*node.data }
                        .try_get_data(&index.cache, node.dim_index)?
                        .map
                        .lookup(&key)
                        .map_or(0, |p| p.len());
                }
                posting_list_lengths.push((*dim_index, length));
            }
        }

        // Sort in descending order of posting list length
        posting_list_lengths.sort_by(|a, b| b.1.cmp(&a.1));

        // Calculate mean and standard deviation of posting list lengths
        // This is used to classify dimensions as common or rare
        let mean_length = if !posting_list_lengths.is_empty() {
            posting_list_lengths
                .iter()
                .map(|(_, len)| *len)
                .sum::<usize>() as f64
                / posting_list_lengths.len() as f64
        } else {
            0.0
        };

        let variance = if !posting_list_lengths.is_empty() {
            posting_list_lengths
                .iter()
                .map(|(_, len)| {
                    let diff = *len as f64 - mean_length;
                    diff * diff
                })
                .sum::<f64>()
                / posting_list_lengths.len() as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();
        let std_dev_threshold = 2.0;
        let rare_threshold = mean_length - (std_dev_threshold * std_dev);

        let mut query_entries = Vec::new();
        for (dim_index, length) in posting_list_lengths {
            if let Some((_, value)) = self
                .query_vector
                .entries
                .iter()
                .find(|(d, _)| *d == dim_index)
            {
                let dim_type = if length > rare_threshold as usize {
                    SparseQueryVectorDimensionType::Common
                } else {
                    SparseQueryVectorDimensionType::Rare
                };
                query_entries.push((dim_index, dim_type, *value));
            }
        }

        Ok(SparseQueryVector::new(
            self.query_vector.vector_id,
            query_entries,
        ))
    }

    pub fn sequential_search(&self, index: &InvertedIndexSparseAnnBasic) -> Vec<SparseAnnResult> {
        let mut dot_products: HashMap<u32, u32> = HashMap::new();

        let mut sorted_query_dims: Vec<(u32, f32)> = self.query_vector.entries.clone();
        sorted_query_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Iterate over the query vector dimensions
        for &(dim_index, dim_value) in &sorted_query_dims {
            if let Some(node) = index.find_node(dim_index) {
                let quantized_query_value = InvertedIndexSparseAnnNodeBasic::quantize(dim_value);
                let start_key: u8 = 63u8;
                let end_key: u8 = match quantized_query_value {
                    0..=15 => 47,
                    16..=31 => 31,
                    32..=47 => 15,
                    _ => 0,
                };
                for key in start_key..=end_key {
                    let arc_rwlock_vec_lazy_item = &node.shared_get().data[key as usize];
                    let p = arc_rwlock_vec_lazy_item.read().unwrap();
                    for lazy_item_u32 in p.iter() {
                        let vector_id = lazy_item_u32.get_data(index.cache.clone());
                        let dot_product = dot_products.entry(*vector_id).or_insert(0u32);
                        *dot_product += (quantized_query_value * key) as u32;
                    }
                }
            }
        }

        // Create a min-heap to keep track of the top K results
        let mut heap = BinaryHeap::with_capacity(K + 1);

        // Process the dot products and maintain the top K results
        for (vector_id, sim_score) in dot_products.iter() {
            heap.push(SparseAnnResult {
                vector_id: *vector_id,
                similarity: *sim_score,
            });
            if heap.len() > K {
                heap.pop();
            }
        }

        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<SparseAnnResult> = heap.into_vec();
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results
    }

    pub fn sequential_search_tshashmap(
        &self,
        index: &InvertedIndexSparseAnnBasicTSHashmap,
        // 4, 5, 6
        quantization_bits: u8,
        values_upper_bound: f32,
        k: Option<usize>,
    ) -> Result<Vec<SparseAnnResult>, BufIoError> {
        let sparse_query_vector = self.create_sparse_query_vector(index)?;
        let mut dot_products: HashMap<u32, u32> = HashMap::new();
        // same as `0.5` quantized
        let half_quantized = 1u8 << (quantization_bits - 1);
        // same as `1` quantized
        let one_quantized = ((1u32 << quantization_bits) - 1) as u8;

        let mut sorted_query_dims: Vec<(u32, SparseQueryVectorDimensionType, f32)> =
            sparse_query_vector.entries.clone();
        sorted_query_dims.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

        // Keep a list of ids that are shortlisted for cuckoo filter lookups
        let mut shortlisted_ids: HashSet<u32> = HashSet::new();

        // Iterate over the query vector dimensions
        for &(dim_index, dimension_type, dim_value) in &sorted_query_dims {
            if let Some(node) = index.find_node(dim_index) {
                let quantized_query_value = node.quantize(dim_value, values_upper_bound) as u32;

                let is_low = quantized_query_value < half_quantized as u32;

                match (dimension_type, is_low) {
                    (SparseQueryVectorDimensionType::Common, true) => {
                        // Common and low quantized value
                        // Only read the values from the cuckoo filter tree, without
                        // going through the entire list of values for this dimension.
                        for vector_id in shortlisted_ids.iter() {
                            if let Some(index) = node.find_key_of_id(*vector_id, &index.cache)? {
                                let dot_product = dot_products.entry(*vector_id).or_insert(0u32);
                                *dot_product += quantized_query_value * index as u32;
                            }
                        }
                    }
                    (SparseQueryVectorDimensionType::Rare, true)
                    | (SparseQueryVectorDimensionType::Common, false) => {
                        // "Rare and low quantized value" or "Common and high quantized value"
                        // Iterate through the map/list ONLY for until a certain threshold
                        // (say 3/4th) of quantized keys (i.e. 48..64). Include these ids in
                        // the cuckoo filter lookup list for other dimensions. Also do
                        // cuckoo filter lookups using the shortlisted ids (from other
                        // dimensions) on this dimension, ensure we do NOT double count
                        // the vectorids that were already encountered in the iteration
                        // from 48..64

                        let mut new_ids = HashSet::new();

                        // First iterate through the map/list
                        for key in (half_quantized..=one_quantized).rev() {
                            let mut current_versioned_pagepool = unsafe { &*node.data }
                                .try_get_data(&index.cache, node.dim_index)?
                                .map
                                .lookup(&key);
                            while let Some(versioned_pagepool) = current_versioned_pagepool {
                                for x in versioned_pagepool.pagepool.inner.iter() {
                                    for x in x.iter() {
                                        let vec_id = *x;

                                        let dot_product =
                                            dot_products.entry(vec_id).or_insert(0u32);
                                        *dot_product += quantized_query_value * key as u32;

                                        new_ids.insert(vec_id);
                                    }
                                }
                                current_versioned_pagepool =
                                    versioned_pagepool.next.read().unwrap().clone();
                            }
                        }

                        // Then do cuckoo filter lookups using the shortlisted ids
                        for vector_id in shortlisted_ids.iter() {
                            if new_ids.contains(vector_id) {
                                continue;
                            }
                            if let Some(index) = node.find_key_of_id(*vector_id, &index.cache)? {
                                let dot_product = dot_products.entry(*vector_id).or_insert(0u32);
                                *dot_product += quantized_query_value * index as u32;
                            }
                        }

                        shortlisted_ids.extend(new_ids);
                    }
                    (SparseQueryVectorDimensionType::Rare, false) => {
                        // Rare and high quantized value
                        // Iterate through the full list of values for this dimension
                        // in inverted index, and then use these shortlisted ids to lookup
                        // when needed in the cuckoo filter tree of the other dims
                        for key in (0..=one_quantized).rev() {
                            let mut current_versioned_pagepool = unsafe { &*node.data }
                                .try_get_data(&index.cache, node.dim_index)?
                                .map
                                .lookup(&key);
                            while let Some(versioned_pagepool) = current_versioned_pagepool {
                                for x in versioned_pagepool.pagepool.inner.iter() {
                                    for x in x.iter() {
                                        let vec_id = *x;
                                        let dot_product =
                                            dot_products.entry(vec_id).or_insert(0u32);
                                        *dot_product += quantized_query_value * key as u32;

                                        // Shortlist vector id for future cuckoo filter lookups
                                        shortlisted_ids.insert(vec_id);
                                    }
                                }
                                current_versioned_pagepool =
                                    versioned_pagepool.next.read().unwrap().clone();
                            }
                        }
                    }
                }
            }
        }

        // Create a min-heap to keep track of the top K results
        let mut heap = BinaryHeap::with_capacity(k.map_or(dot_products.len(), |k| k * 100) + 1);

        // Process the dot products and maintain the top K results
        for (vector_id, similarity) in dot_products.into_iter() {
            heap.push(SparseAnnResult {
                vector_id,
                similarity,
            });
            if let Some(k) = k {
                if heap.len() > k * 100 {
                    heap.pop();
                }
            }
        }

        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<SparseAnnResult> = heap.into_vec();
        results.sort_by(|a, b| b.similarity.cmp(&a.similarity));
        Ok(results)
    }

    pub fn sequential_search_dashmap(
        &self,
        index: &InvertedIndexSparseAnnBasicDashMap,
    ) -> Vec<SparseAnnResult> {
        let mut dot_products: HashMap<u32, u32> = HashMap::new();

        let mut sorted_query_dims: Vec<(u32, f32)> = self.query_vector.entries.clone();
        sorted_query_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Iterate over the query vector dimensions
        for &(dim_index, dim_value) in &sorted_query_dims {
            if let Some(node) = index.find_node(dim_index) {
                let quantized_query_value =
                    InvertedIndexSparseAnnNodeBasicDashMap::quantize(dim_value);
                let start_key: u8 = 63u8;
                let end_key: u8 = match quantized_query_value {
                    0..=15 => 47,
                    16..=31 => 31,
                    32..=47 => 15,
                    _ => 0,
                };
                for key in start_key..=end_key {
                    for x in node.shared_get().data.clone() {
                        if x.1 == key {
                            let vec_id = x.0;
                            let dot_product = dot_products.entry(vec_id).or_insert(0u32);
                            *dot_product += (quantized_query_value * key) as u32;
                        }
                    }
                }
            }
        }

        // Create a min-heap to keep track of the top K results
        let mut heap = BinaryHeap::with_capacity(K + 1);

        // Process the dot products and maintain the top K results
        for (vector_id, sim_score) in dot_products.iter() {
            heap.push(SparseAnnResult {
                vector_id: *vector_id,
                similarity: *sim_score,
            });
            if heap.len() > K {
                heap.pop();
            }
        }

        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<SparseAnnResult> = heap.into_vec();
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results
    }
}
