use crate::models::identity_collections::IdentityMapKey;
use crate::models::types::SparseVector;
use dashmap::DashMap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
// use std::sync::Mutex;

use super::inverted_index_old::InvertedIndex;

const K: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct KNNResult {
    vector_id: IdentityMapKey,
    similarity: f32,
}

impl Eq for KNNResult {}

impl PartialOrd for KNNResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}

impl Ord for KNNResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct KNNQuery {
    /// Query vector is a pair of non-zero values and its dimension
    query_vector: SparseVector,
}

impl KNNQuery {
    pub fn new(query_vector: SparseVector) -> Self {
        KNNQuery { query_vector }
    }

    pub fn concurrent_search(&self, index: &InvertedIndex<f32>) -> Vec<KNNResult> {
        let dot_products: DashMap<IdentityMapKey, f32> = DashMap::new();

        // Iterate over the query vector dimensions concurrently with rayon
        self.query_vector
            .entries
            .par_iter()
            .for_each(|&(dim_index, dim_value)| {
                if let Some(node) = index.find_node(dim_index) {
                    // Get all data for this dimension
                    let data: &DashMap<IdentityMapKey, crate::models::lazy_load::LazyItem<f32>> =
                        &node.data;

                    // Iterate over the DashMap concurrently
                    data.iter().par_bridge().for_each(|entry| {
                        let vector_id = entry.key();
                        let value = entry.value().get_data(index.cache.clone());
                        let mut dot_product = dot_products.entry(vector_id.clone()).or_insert(0.0);
                        *dot_product += dim_value * *value;
                    });
                }
            });

        // Create a min-heap within a Mutex to keep track of the top K results concurrently
        // let heap = Mutex::new(BinaryHeap::with_capacity(K + 1));
        // Do not use mutex and concurrency to avoid bottlenecks while pushing result
        let mut heap = BinaryHeap::with_capacity(K + 1);

        // Process the dot products and maintain the top K results, Pushing dot_products into heap.
        dot_products.iter().for_each(|x| {
            // let mut heap = heap.lock().expect("Panic due to lock over heap value");
            heap.push(KNNResult {
                vector_id: x.key().clone(),
                similarity: *x.value(),
            });
            if heap.len() > K {
                heap.pop();
            }
        });

        // let mut results = heap.lock().unwrap().clone().into_vec();
        let mut results = heap.clone().into_vec();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results
    }

    pub fn sequential_search(&self, index: &InvertedIndex<f32>) -> Vec<KNNResult> {
        let mut dot_products: HashMap<IdentityMapKey, f32> = HashMap::new();

        // Iterate over the query vector dimensions
        for &(dim_index, dim_value) in &self.query_vector.entries {
            if let Some(node) = index.find_node(dim_index) {
                // Get all data for this dimension
                let data: &DashMap<IdentityMapKey, crate::models::lazy_load::LazyItem<f32>> =
                    &node.data;

                // Iterate over the DashMap
                for entry in data.iter() {
                    let vector_id = entry.key();
                    let value = entry.value().get_data(index.cache.clone());
                    let dot_product = dot_products.entry(vector_id.clone()).or_insert(0.0);
                    *dot_product += dim_value * *value;
                }
            }
        }

        // Create a min-heap to keep track of the top K results
        let mut heap = BinaryHeap::with_capacity(K + 1);

        // Process the dot products and maintain the top K results
        for (vector_id, sim_score) in dot_products.iter() {
            heap.push(KNNResult {
                vector_id: vector_id.clone(),
                similarity: *sim_score,
            });
            if heap.len() > K {
                heap.pop();
            }
        }

        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<KNNResult> = heap.into_vec();
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results
    }

    //Brute search of Knn without using Inverted_Index datastructure and No concurrency
    pub fn brute_search(&self, sparse_vec: Vec<SparseVector>) -> Vec<KNNResult> {
        let mut dot_products: HashMap<u32, f32> = HashMap::new(); //(vector_id, similarity)

        let _ = &self
            .query_vector
            .entries
            .iter()
            .for_each(|&(dim_index, dim_value)| {
                sparse_vec.iter().for_each(|svec| {
                    svec.entries.iter().for_each(|&(x, y)| {
                        if x == dim_index {
                            let dot_product = dot_products.entry(svec.vector_id).or_insert(0.0);
                            *dot_product += dim_value * y;
                        }
                    });
                });
            });

        // Create a min-heap to keep track of the top K results
        let mut heap: BinaryHeap<KNNResult> = BinaryHeap::with_capacity(K + 1);

        // Process the dot products and maintain the top K results
        for (vector_id, sim_score) in dot_products.iter() {
            heap.push(KNNResult {
                vector_id: IdentityMapKey::Int(*vector_id),
                similarity: *sim_score,
            });
            if heap.len() > K {
                heap.pop();
            }
        }
        // Convert the heap to a vector and reverse it to get descending order
        let mut results: Vec<KNNResult> = heap.into_vec();
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::types::SparseVector;

    #[test]
    fn test_knn_search() {
        let index = InvertedIndex::new();

        // Add some test vectors
        let vectors = vec![
            SparseVector::new(1, vec![(0, 1.0), (2, 2.0), (4, 3.0)]),
            SparseVector::new(2, vec![(1, 1.5), (3, 2.5), (5, 3.5)]),
            SparseVector::new(3, vec![(0, 0.5), (2, 1.5), (4, 2.5)]),
            SparseVector::new(4, vec![(1, 2.0), (3, 3.0), (5, 4.0)]),
        ];
        let vectors_brute = vectors.clone();

        for vector in vectors {
            index.add_sparse_vector(vector).unwrap();
        }

        // Create a query vector
        let query_vector = vec![(0, 1.0), (2, 2.0), (4, 3.0)];
        let knn_query = KNNQuery::new(SparseVector {
            vector_id: 0,
            entries: query_vector,
        });

        // Perform the KNN search sequentially
        let results = knn_query.sequential_search(&index);

        // Check the results
        assert_eq!(results.len(), 2); // We expect 2 results because only 3 vectors have matching dimensions
        assert_eq!(results[0].vector_id, IdentityMapKey::Int(1)); // Vector 1 should be the most similar
        assert_eq!(results[1].vector_id, IdentityMapKey::Int(3)); // Vector 3 should be the second most similar
        assert!(results[0].similarity > results[1].similarity);

        // Perform the KNN search with concurrency
        let results_concurrent = knn_query.concurrent_search(&index);

        assert_eq!(results_concurrent.len(), 2); // We expect 2 results because only 3 vectors have matching dimensions
        assert_eq!(results_concurrent[0].vector_id, IdentityMapKey::Int(1)); // Vector 1 should be the most similar
        assert_eq!(results_concurrent[1].vector_id, IdentityMapKey::Int(3)); // Vector 3 should be the second most similar
        assert!(results_concurrent[0].similarity > results_concurrent[1].similarity);

        // Perform the KNN brute_search without Inverted_Index and No concurrency
        let results_brute = knn_query.brute_search(vectors_brute);

        assert_eq!(results_brute.len(), 2); // We expect 2 results because only 3 vectors have matching dimensions
        assert_eq!(results_brute[0].vector_id, IdentityMapKey::Int(1)); // Vector 1 should be the most similar
        assert_eq!(results_brute[1].vector_id, IdentityMapKey::Int(3)); // Vector 3 should be the second most similar
        assert!(results_brute[0].similarity > results_brute[1].similarity);
    }
}
