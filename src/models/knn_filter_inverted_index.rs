use crate::storage::inverted_index::{self, InvertedIndex, InvertedIndexItem};
use std::cmp::Ordering::Equal;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Mul;

use super::lazy_load::LazyItem;

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
                    self_value.get_lazy_data().and_then(|mut self_data| {
                        neighbour
                            .data
                            .get(identity_map_key)
                            .and_then(|neighbour_value| {
                                if neighbour_value.is_valid() {
                                    neighbour_value.get_lazy_data().map(|mut neighbour_data| {
                                        (self_data.get().clone().into() as f32)
                                            * (neighbour_data.get().clone().into() as f32)
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

    // Function to find k-nearest neighbors sorted by distance(dot product) for an InvertedIndex
    pub fn find_k_nearest_neighbours(
        &self,
        k: usize,
        inverted_index: &mut InvertedIndex<T>,
    ) -> Result<Vec<ScoredCandidate>, String> {
        //Sending the root inverted_index_item from which its children can be fetched recursively using traverse_inverted_index function
        let all_items_in_inverted_index = Self::traverse_inverted_index(inverted_index.root.get());

        if k > all_items_in_inverted_index.len() {
            return Err(String::from(
                "K value sent is greater than the number of neighbours!, Please reduce k number",
            ));
        }

        let mut scored_candidates: Vec<ScoredCandidate> = all_items_in_inverted_index
            .iter()
            .map(|item| ScoredCandidate(self.calculate_distance(item), item.dim_index))
            .collect();
        scored_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Equal));
        Ok(scored_candidates.into_iter().take(k).collect())
    }

    // Traverses entire [InvertedIndex] and returns all root, children which are explicit
    pub fn traverse_inverted_index(
        inverted_index_item: &InvertedIndexItem<T>, //root of InvertedIndex is sent first.
    ) -> Vec<InvertedIndexItem<T>> {
        let mut vec_inverted_index_items: Vec<InvertedIndexItem<T>> = Vec::new();

        if !inverted_index_item.implicit {
            // Collecting each inverted_index_item which is explicit to return in a vec
            vec_inverted_index_items.push(inverted_index_item.clone());
            let mut map = inverted_index_item.lazy_children.items.arcshift.clone();
            for (_, child) in map.get().iter() {
                if let LazyItem::Valid { data: Some(x), .. } = child {
                    vec_inverted_index_items.extend(
                        // Calling function recursively for child inverted_index_items
                        Self::traverse_inverted_index(x.clone().get()),
                    );
                }
            }
        }
        vec_inverted_index_items
    }
}
