use std::collections::HashMap;
use std::{cmp::Ordering, collections::BinaryHeap};

use crate::models::types::SparseVector;

use super::inverted_index_sparse_ann::InvertedIndexSparseAnnNode;
use super::inverted_index_sparse_ann_new_ds::InvertedIndexSparseAnnNewDS;

const K: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct SparseAnnResult {
    vector_id: u32,
    similarity: u32,
}

impl Eq for SparseAnnResult {}

impl PartialOrd for SparseAnnResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}

impl Ord for SparseAnnResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct SparseAnnQueryNewDS {
    /// Query vector is a pair of non-zero values and its dimension
    query_vector: SparseVector,
}

impl SparseAnnQueryNewDS {
    pub fn new(query_vector: SparseVector) -> Self {
        SparseAnnQueryNewDS { query_vector }
    }

    pub fn sequential_search(&self, index: &InvertedIndexSparseAnnNewDS) -> Vec<SparseAnnResult> {
        let mut dot_products: HashMap<u32, u32> = HashMap::new();

        let mut sorted_query_dims: Vec<(u32, f32)> = self.query_vector.entries.clone();
        sorted_query_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Iterate over the query vector dimensions
        for &(dim_index, dim_value) in &sorted_query_dims {
            if let Some(node) = index.find_node(dim_index) {
                let quantized_query_value = InvertedIndexSparseAnnNode::quantize(dim_value);
                let start_key: u8 = 63u8;
                let end_key: u8 = match quantized_query_value {
                    0..=15 => 47,
                    16..=31 => 31,
                    32..=47 => 15,
                    _ => 0,
                };
                for key in start_key..=end_key {
                    let growable_data = &node.shared_get().data[key as usize];

                    growable_data.items.iter().for_each(|x| {
                        let res = x.get_data(index.cache.clone());
                        let vector_data = (*res).clone().get().clone();
                        vector_data.data.iter().for_each(|vec_id| {
                            let dot_product = dot_products.entry(*vec_id).or_insert(0u32);
                            *dot_product += (quantized_query_value * key) as u32;
                        });
                    });
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
