use crate::models::identity_collections::IdentityMapKey;
use crate::models::types::SparseVector;
use dashmap::DashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::inverted_index::InvertedIndex;

const K: usize = 5;

#[derive(Debug, Clone, PartialEq)]
struct KNNResult {
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

struct KNNQuery {
    /// Query vector is a pair of non-zero values and its dimension
    query_vector: SparseVector,
}

impl KNNQuery {
    pub fn new(query_vector: SparseVector) -> Self {
        KNNQuery { query_vector }
    }

    pub fn search(&self, index: &InvertedIndex<f32>) -> Vec<KNNResult> {
        let mut dot_products: HashMap<IdentityMapKey, f32> = HashMap::new();

        // Iterate over the query vector dimensions
        for &(dim_index, dim_value) in &self.query_vector.entries {
            if let Some(node) = index.find_node(dim_index) {
                // Get all data for this dimension
                let data: &DashMap<IdentityMapKey, crate::models::lazy_load::LazyItem<f32>> =
                    &node.shared_get().data;

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

        for vector in vectors {
            index.add_sparse_vector(vector).unwrap();
        }

        // Create a query vector
        let query_vector = vec![(0, 1.0), (2, 2.0), (4, 3.0)];
        let knn_query = KNNQuery::new(SparseVector {
            vector_id: 0,
            entries: query_vector,
        });

        // Perform the KNN search
        let results = knn_query.search(&index);

        // Check the results
        assert_eq!(results.len(), 2); // We expect 2 results because only 3 vectors have matching dimensions
        assert_eq!(results[0].vector_id, IdentityMapKey::Int(1)); // Vector 1 should be the most similar
        assert_eq!(results[1].vector_id, IdentityMapKey::Int(3)); // Vector 3 should be the second most similar
        assert!(results[0].similarity > results[1].similarity);
    }
}
