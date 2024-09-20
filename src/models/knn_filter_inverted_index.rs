use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::ops::Mul;

use crate::storage::inverted_index::InvertedIndexItem;

struct ScoredCandidate(f32, u32); // (score, candidate_id)

impl ScoredCandidate {
    fn new(score: f32, candidate_id: u32) -> Self {
        ScoredCandidate(score, candidate_id)
    }
}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for ScoredCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ScoredCandidate {}

fn maintain_top_k(
    heap: &mut BinaryHeap<Reverse<ScoredCandidate>>,
    score: f32,
    candidate_id: u32,
    k: usize,
) {
    if heap.len() < k {
        heap.push(Reverse(ScoredCandidate(score, candidate_id)));
    } else if score > heap.peek().unwrap().0 .0 {
        heap.pop();
        heap.push(Reverse(ScoredCandidate(score, candidate_id)));
    }
}

impl<T> InvertedIndexItem<T>
where
    T: Clone + 'static + Mul<Output = T> + Into<f32>,
{
    // Calculate distance using dot product
    fn calculate_distance(&self, neighbour: &InvertedIndexItem<T>) -> f32 {
        let self_inverted_index_data = self.data.iter();
        let mut vec_map_self = HashMap::new();
        let mut vec_map_neighbour = HashMap::new();

        for xmap in self_inverted_index_data {
            let vector_id = xmap.key().clone();
            let vec_id_neighbour = vector_id.clone();
            if xmap.value().is_valid() {
                if let Some(value) = xmap.value().get_lazy_data() {
                    vec_map_self.insert(vector_id, value);
                }
            }

            if let Some(res) = neighbour.data.get(&vec_id_neighbour) {
                if res.is_valid() {
                    if let Some(val) = res.get_lazy_data() {
                        vec_map_neighbour.insert(vec_id_neighbour, val);
                    }
                }
            };
        }
        // Calculating the dot product of vec_map_self and vec_map_neighbour
        let mut dot_product = 0.0;
        for (key, value_self) in &vec_map_self {
            if let Some(value_neighbour) = vec_map_neighbour.get(key) {
                // Perform the multiplication and accumulate the dot product
                dot_product += (value_self.clone().try_into_inner().unwrap().into())
                    * (value_neighbour.clone()).try_into_inner().unwrap().into();
            }
        }

        dot_product
    }

    // Function to find k-nearest neighbors sorted by distance(dot product) for array of InvertedIndex items.
    fn find_k_nearest_neighbors(
        &self,
        k: usize,
        items: &[InvertedIndexItem<T>],
    ) -> Result<Vec<ScoredCandidate>, String> {
        if k > items.len() {
            return Err(String::from("K value sent is greater than the number of neighbours, Please check!"));
        }
        let mut scored_candidates: Vec<ScoredCandidate> = items
            .iter()
            .map(|item| ScoredCandidate(self.calculate_distance(item), item.dim_index))
            .collect();
        scored_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut res: Vec<ScoredCandidate> = Vec::new();
        let _ = scored_candidates.into_iter().take(k).map(|x| res.push(x));
        Ok(res)
    }
}
