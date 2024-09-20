use crate::storage::inverted_index::InvertedIndexItem;
use std::cmp::Ordering::Equal;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Mul;

pub struct ScoredCandidate(f32, u32); // (score, candidate_id)

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
    // Calculate distance using dot product between self and InvertedIndexItem
    pub fn calculate_distance(&self, neighbour: &InvertedIndexItem<T>) -> f32 {
        let res: Vec<Option<f32>> = self
            .data
            .iter()
            .map(|self_value| {
                let identity_map_key = self_value.key();
                if self_value.is_valid() {
                    self_value.get_lazy_data().and_then(|self_data| {
                        neighbour
                            .data
                            .get(identity_map_key)
                            .and_then(|neighbour_value| {
                                if neighbour_value.is_valid() {
                                    neighbour_value.get_lazy_data().map(|neighbour_data| {
                                        (self_data.clone().try_into_inner().unwrap().into() as f32)
                                            * (neighbour_data
                                                .clone()
                                                .try_into_inner()
                                                .unwrap()
                                                .into()
                                                as f32)
                                    })
                                } else {
                                    None
                                }
                            })
                    })
                } else {
                    None
                }
            })
            .collect();
        let mut dot_product = 0.0;
        let _ = res.iter().map(|x| {
            if let Some(i) = x {
                dot_product += i;
            }
        });
        dot_product
    }

    // Function to find k-nearest neighbors sorted by distance(dot product) for array of InvertedIndex items.
    pub fn find_k_nearest_neighbours(
        &self,
        k: usize,
        items: &[InvertedIndexItem<T>],
    ) -> Result<Vec<ScoredCandidate>, String> {
        if k > items.len() {
            return Err(String::from(
                "K value sent is greater than the number of neighbours, Please check!",
            ));
        }
        let mut scored_candidates: Vec<ScoredCandidate> = items
            .iter()
            .map(|item| ScoredCandidate(self.calculate_distance(item), item.dim_index))
            .collect();
        scored_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Equal));
        Ok(scored_candidates.into_iter().take(k).collect())
    }
}
