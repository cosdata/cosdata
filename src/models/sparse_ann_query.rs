use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::models::buffered_io::BufIoError;

use crate::models::types::SparseVector;
use crate::models::versioned_vec::VersionedVec;
use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter::Peekable;
use std::sync::RwLockReadGuard;

use super::inverted_index::InvertedIndexRoot;
use super::tf_idf_index::{TFIDFIndexRoot, TermQuotient};
use super::versioned_vec::VersionedVecIter;

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
                    unsafe { &*node.data }
                        .try_get_data(&index.cache)?
                        .map
                        .with_value(&key, |vec| {
                            for vec_id in vec.iter() {
                                let dot_product = dot_products.entry(vec_id).or_insert(0u32);
                                *dot_product += quantized_query_value * key as u32;
                            }
                        });
                }
            } else {
                // Low quantized value
                // Iterate through the map/list ONLY for until a certain threshold (say 3/4th)
                // of quantized keys (i.e. 48..64 for 6 bit quantization).

                for key in (early_terminate_value..=one_quantized).rev() {
                    unsafe { &*node.data }
                        .try_get_data(&index.cache)?
                        .map
                        .with_value(&key, |vec| {
                            for vec_id in vec.iter() {
                                let dot_product = dot_products.entry(vec_id).or_insert(0u32);
                                *dot_product += quantized_query_value * key as u32;
                            }
                        });
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
        index: &TFIDFIndexRoot,
        k: Option<usize>,
    ) -> Result<Vec<SparseAnnIDFResult>, BufIoError> {
        const BUCKETS: usize = 512;
        let documents_count = index
            .total_documents_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let mut heads = BinaryHeap::new();
        let mut locks = Vec::new();

        for (term_hash, _) in self.query_vector.entries {
            let dim_index = term_hash & (u16::MAX as u32);
            let quotient = (term_hash >> 16) as TermQuotient;
            if let Some(node) = index.find_node(dim_index) {
                let data = unsafe { &*node.data }.try_get_data(&index.cache)?;
                if let Some(term) = data.map.lookup(&quotient) {
                    let documents = term.documents.read().unwrap();
                    let idf = get_idf(documents_count, documents.len() as u32);

                    let head = PostingListHead::new(&documents, idf);
                    locks.push(unsafe {
                        std::mem::transmute::<
                            RwLockReadGuard<'_, VersionedVec<(u32, f32)>>,
                            RwLockReadGuard<'_, VersionedVec<(u32, f32)>>,
                        >(documents)
                    });
                    heads.push(head);
                }
            }
        }

        let mut buckets = [(u32::MAX, f32::NEG_INFINITY); BUCKETS];

        while let Some(mut head) = heads.pop() {
            let Some((doc_id, tf)) = head.pop() else {
                continue;
            };

            let mut score = tf * head.idf;

            if head.peek().is_some() {
                heads.push(head);
            }

            while let Some(head) = heads.peek() {
                let Some((doc_id1, tf)) = head.peek() else {
                    heads.pop();
                    continue;
                };
                if doc_id1 != &doc_id {
                    break;
                }

                score += tf * head.idf;
                let mut head = heads.pop().unwrap();
                head.pop();
                if head.peek().is_some() {
                    heads.push(head);
                }
            }

            let index = doc_id as usize % BUCKETS;
            if score > buckets[index].1 {
                buckets[index] = (doc_id, score);
            }
        }

        let mut results: Vec<_> = buckets
            .into_iter()
            .filter(|bucket| bucket.0 != u32::MAX)
            .map(|(id, score)| SparseAnnIDFResult {
                document_id: id,
                score,
            })
            .collect();

        results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        if let Some(k) = k {
            results.truncate(k);
        }

        Ok(results)
    }
}

struct PostingListHead<'a> {
    iter: UnsafeCell<Peekable<VersionedVecIter<'a, (u32, f32)>>>,
    pub idf: f32,
}

impl<'a> PostingListHead<'a> {
    pub fn new(documents: &VersionedVec<(u32, f32)>, idf: f32) -> Self {
        Self {
            iter: UnsafeCell::new(
                unsafe {
                    std::mem::transmute::<&VersionedVec<(u32, f32)>, &'a VersionedVec<(u32, f32)>>(
                        documents,
                    )
                }
                .iter()
                .peekable(),
            ),
            idf,
        }
    }

    pub fn pop(&mut self) -> Option<(u32, f32)> {
        self.iter.get_mut().next()
    }

    pub fn peek(&self) -> Option<&(u32, f32)> {
        unsafe { &mut *self.iter.get() }.peek()
    }
}

impl Ord for PostingListHead<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let Some(s) = self.peek() else { unreachable!() };

        let Some(o) = other.peek() else {
            unreachable!()
        };

        o.0.cmp(&s.0)
    }
}

impl PartialOrd for PostingListHead<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PostingListHead<'_> {}

impl PartialEq for PostingListHead<'_> {
    fn eq(&self, other: &Self) -> bool {
        let Some(s) = self.peek() else { unreachable!() };

        let Some(o) = other.peek() else {
            unreachable!()
        };

        s.0.eq(&o.0)
    }
}

fn get_idf(documents_count: u32, documents_containing_term: u32) -> f32 {
    (((documents_count - documents_containing_term) as f32 + 0.5)
        / (documents_containing_term as f32 + 0.5))
        .ln_1p()
}
