use dashmap::DashMap;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use std::collections::HashMap;
use std::{cmp::Ordering, collections::BinaryHeap};

use crate::models::lazy_load::LazyItemVec;
use crate::models::types::SparseVector;

use super::inverted_index_new::{InvertedIndexNew, InvertedIndexNode};

const K: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct KNNResultNew {
    vector_id: u32,
    similarity: f32,
}

impl Eq for KNNResultNew {}

impl PartialOrd for KNNResultNew {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}

impl Ord for KNNResultNew {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct KNNQueryNew {
    /// Query vector is a pair of non-zero values and its dimension
    query_vector: SparseVector,
}

impl KNNQueryNew {
    pub fn new(query_vector: SparseVector) -> Self {
        KNNQueryNew { query_vector }
    }

    pub fn concurrent_search(&self, index: &InvertedIndexNew) -> Vec<KNNResultNew> {
        let dot_products: DashMap<u32, f32> = DashMap::new();

        let mut sorted_query_dims: Vec<(u32, f32)> = self.query_vector.entries.clone();
        sorted_query_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        sorted_query_dims
            .par_iter()
            .for_each(|&(dim_index, dim_value)| {
                if let Some(node) = index.find_node(dim_index) {
                    let quantized_query_value = InvertedIndexNode::quantize(dim_value);
                    let start_key = 63u8;
                    let end_key = match quantized_query_value {
                        0..=15 => 47,
                        16..=31 => 31,
                        32..=47 => 15,
                        _ => 0,
                    };
                    for key in start_key..=end_key {
                        let lazy_item_vec: &LazyItemVec<u32> =
                            &node.shared_get().data[key as usize];
                        if !lazy_item_vec.is_empty() {
                            for lazy_item in lazy_item_vec.iter() {
                                let vector_id = lazy_item.get_data(index.cache.clone());
                                let mut dot_product = dot_products.entry(*vector_id).or_insert(0.0);
                                *dot_product += dim_value * (key as f32 / 63.0);
                            }
                        }
                    }
                }
            });

        let mut heap = BinaryHeap::with_capacity(K + 1);

        dot_products.iter().for_each(|x| {
            heap.push(KNNResultNew {
                vector_id: *x.key(),
                similarity: *x.value(),
            });
            if heap.len() > K {
                heap.pop();
            }
        });

        let mut results = heap.into_vec();
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results
    }

    pub fn sequential_search(&self, index: &InvertedIndexNew) -> Vec<KNNResultNew> {
        let mut dot_products: HashMap<u32, f32> = HashMap::new();

        // Iterate over the query vector dimensions
        for &(dim_index, dim_value) in &self.query_vector.entries {
            if let Some(node) = index.find_node(dim_index) {
                let quantized_query_value = InvertedIndexNode::quantize(dim_value);
                let start_key = 63u8;
                let end_key = match quantized_query_value {
                    0..=15 => 47,
                    16..=31 => 31,
                    32..=47 => 15,
                    _ => 0,
                };
                for key in start_key..=end_key {
                    let lazy_item_vec: &LazyItemVec<u32> = &node.shared_get().data[key as usize];
                    if !lazy_item_vec.is_empty() {
                        for lazy_item in lazy_item_vec.iter() {
                            let vector_id = lazy_item.get_data(index.cache.clone());
                            let mut dot_product = dot_products.entry(*vector_id).or_insert(0.0);
                            *dot_product += dim_value * (key as f32 / 63.0);
                        }
                    }
                }
            }
        }

        // Create a min-heap to keep track of the top K results
        let mut heap = BinaryHeap::with_capacity(K + 1);

        // Process the dot products and maintain the top K results
        for (vector_id, sim_score) in dot_products.iter() {
            heap.push(KNNResultNew {
                vector_id: *vector_id,
                similarity: *sim_score,
            });
            if heap.len() > K {
                heap.pop();
            }
        }

        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<KNNResultNew> = heap.into_vec();
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results
    }
}
