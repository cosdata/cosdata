pub(crate) mod types;
use super::{IndexOps, InternalSearchResult};
use crate::{
    config_loader::Config,
    models::{
        buffered_io::BufIoError,
        collection::{Collection, RawVectorEmbedding},
        common::WaCustomError,
        inverted_index::InvertedIndexRoot,
        meta_persist::store_values_upper_bound,
        sparse_ann_query::{SparseAnnQueryBasic, SparseAnnResult},
        types::{InternalId, MetaDb, SparseVector},
        versioning::VersionNumber,
    },
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        RwLock,
    },
};
use types::{SamplingData, SparsePair};

pub struct SparseInputEmbedding(pub InternalId, pub Vec<SparsePair>);

pub struct SparseSearchInput(pub Vec<SparsePair>);

pub struct SparseSearchOptions {
    pub top_k: Option<usize>,
    pub early_terminate_threshold: Option<f32>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct InvertedIndexData {
    pub quantization_bits: u8,
    pub sample_threshold: usize,
}

pub struct InvertedIndex {
    pub root: InvertedIndexRoot,
    pub values_upper_bound: RwLock<f32>,
    pub is_configured: AtomicBool,
    pub sampling_data: SamplingData,
    pub vectors: RwLock<Vec<SparseInputEmbedding>>,
    pub vectors_collected: AtomicUsize,
    pub sample_threshold: usize,
}

unsafe impl Send for InvertedIndex {}
unsafe impl Sync for InvertedIndex {}

impl InvertedIndex {
    pub fn new(
        root_path: PathBuf,
        quantization_bits: u8,
        sample_threshold: usize,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexRoot::new(root_path, quantization_bits)?;

        Ok(Self {
            root,
            values_upper_bound: RwLock::new(1.0),
            is_configured: AtomicBool::new(false),
            sampling_data: SamplingData::default(),
            vectors: RwLock::new(Vec::new()),
            vectors_collected: AtomicUsize::new(0),
            sample_threshold,
        })
    }

    pub fn insert(
        &self,
        id: InternalId,
        pairs: Vec<SparsePair>,
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        let id = id.into();
        for pair in &pairs {
            self.root.insert(
                pair.0,
                pair.1,
                id,
                version,
                *self.values_upper_bound.read().unwrap(),
            )?;
        }
        Ok(())
    }

    pub fn mark_embedding_as_deleted(
        &self,
        id: InternalId,
        pairs: &[SparsePair],
        version: VersionNumber,
    ) -> Result<(), BufIoError> {
        let id = id.into();
        for pair in pairs {
            self.root.delete(
                pair.0,
                pair.1,
                id,
                version,
                *self.values_upper_bound.read().unwrap(),
            )?;
        }
        Ok(())
    }
}

impl IndexOps for InvertedIndex {
    type IndexingInput = SparseInputEmbedding;
    type SearchInput = SparseSearchInput;
    type SearchOptions = SparseSearchOptions;
    type Data = InvertedIndexData;

    fn validate_embedding(&self, _embedding: Self::IndexingInput) -> Result<(), WaCustomError> {
        Ok(())
    }

    fn index_embeddings(
        &self,
        _collection: &Collection,
        embeddings: Vec<Self::IndexingInput>,
        version: VersionNumber,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        embeddings
            .into_iter()
            .try_for_each(|SparseInputEmbedding(id, pairs)| self.insert(id, pairs, version))?;
        Ok(())
    }

    fn delete_embedding(
        &self,
        id: InternalId,
        raw_emb: &RawVectorEmbedding,
        version: VersionNumber,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        let Some(pairs) = &raw_emb.sparse_values else {
            return Ok(());
        };
        self.mark_embedding_as_deleted(id, pairs, version)?;
        Ok(())
    }

    fn sample_embedding(&self, embedding: &Self::IndexingInput) {
        for pair in &embedding.1 {
            let value = pair.1;

            if value > 1.0 {
                self.sampling_data.above_1.fetch_add(1, Ordering::Relaxed);
            }

            if value > 2.0 {
                self.sampling_data.above_2.fetch_add(1, Ordering::Relaxed);
            }

            if value > 3.0 {
                self.sampling_data.above_3.fetch_add(1, Ordering::Relaxed);
            }

            if value > 4.0 {
                self.sampling_data.above_4.fetch_add(1, Ordering::Relaxed);
            }

            if value > 5.0 {
                self.sampling_data.above_5.fetch_add(1, Ordering::Relaxed);
            }

            if value > 6.0 {
                self.sampling_data.above_6.fetch_add(1, Ordering::Relaxed);
            }

            if value > 7.0 {
                self.sampling_data.above_7.fetch_add(1, Ordering::Relaxed);
            }

            if value > 8.0 {
                self.sampling_data.above_8.fetch_add(1, Ordering::Relaxed);
            }

            if value > 9.0 {
                self.sampling_data.above_9.fetch_add(1, Ordering::Relaxed);
            }

            self.sampling_data
                .values_collected
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn finalize_sampling(
        &self,
        lmdb: &MetaDb,
        config: &Config,
        _embeddings: &[Self::IndexingInput],
    ) -> Result<(), WaCustomError> {
        let values_count = self.sampling_data.values_collected.load(Ordering::Relaxed) as f32;

        let above_1_percent =
            (self.sampling_data.above_1.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_2_percent =
            (self.sampling_data.above_2.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_3_percent =
            (self.sampling_data.above_3.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_4_percent =
            (self.sampling_data.above_4.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_5_percent =
            (self.sampling_data.above_5.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_6_percent =
            (self.sampling_data.above_6.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_7_percent =
            (self.sampling_data.above_7.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_8_percent =
            (self.sampling_data.above_8.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_9_percent =
            (self.sampling_data.above_9.load(Ordering::Relaxed) as f32 / values_count) * 100.0;

        let values_upper_bound = if above_1_percent <= config.indexing.clamp_margin_percent {
            1.0
        } else if above_2_percent <= config.indexing.clamp_margin_percent {
            2.0
        } else if above_3_percent <= config.indexing.clamp_margin_percent {
            3.0
        } else if above_4_percent <= config.indexing.clamp_margin_percent {
            4.0
        } else if above_5_percent <= config.indexing.clamp_margin_percent {
            5.0
        } else if above_6_percent <= config.indexing.clamp_margin_percent {
            6.0
        } else if above_7_percent <= config.indexing.clamp_margin_percent {
            7.0
        } else if above_8_percent <= config.indexing.clamp_margin_percent {
            8.0
        } else if above_9_percent <= config.indexing.clamp_margin_percent {
            9.0
        } else {
            10.0
        };

        *self.values_upper_bound.write().unwrap() = values_upper_bound;
        self.is_configured.store(true, Ordering::Release);
        store_values_upper_bound(lmdb, values_upper_bound)?;
        Ok(())
    }

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::IndexingInput>> {
        &self.vectors
    }

    fn increment_collected_count(&self, count: usize) -> usize {
        self.vectors_collected.fetch_add(count, Ordering::SeqCst)
    }

    fn sample_threshold(&self) -> usize {
        self.sample_threshold
    }

    fn is_configured(&self) -> bool {
        self.is_configured.load(Ordering::Acquire)
    }

    fn flush(&self, _: &Collection, _version: VersionNumber) -> Result<(), WaCustomError> {
        self.root.serialize()?;
        self.root.cache.flush_all()?;
        Ok(())
    }

    fn get_data(&self) -> Self::Data {
        Self::Data {
            quantization_bits: self.root.root.quantization_bits,
            sample_threshold: self.sample_threshold,
        }
    }

    fn search_internal(
        &self,
        collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<InternalSearchResult>, WaCustomError> {
        let sparse_vec = SparseVector {
            vector_id: u32::MAX,
            entries: query.0.iter().map(|pair| (pair.0, pair.1)).collect(),
        };

        let results = SparseAnnQueryBasic::new(sparse_vec).sequential_search(
            &self.root,
            self.root.root.quantization_bits,
            *self.values_upper_bound.read().unwrap(),
            options
                .early_terminate_threshold
                .unwrap_or(config.search.early_terminate_threshold),
            if config.rerank_sparse_with_raw_values {
                config.sparse_raw_values_reranking_factor
            } else {
                1
            },
            options.top_k,
        )?;

        if config.rerank_sparse_with_raw_values {
            finalize_sparse_ann_results(
                collection,
                results,
                &query.0,
                options.top_k,
                return_raw_text,
            )
        } else {
            Ok(results
                .into_iter()
                .map(|result| {
                    (
                        InternalId::from(result.vector_id),
                        None,
                        None,
                        result.similarity as f32,
                        None,
                    )
                })
                .collect())
        }
    }
}

fn finalize_sparse_ann_results(
    collection: &Collection,
    intermediate_results: Vec<SparseAnnResult>,
    query: &[SparsePair],
    k: Option<usize>,
    return_raw_text: bool,
) -> Result<Vec<InternalSearchResult>, WaCustomError> {
    let mut results = Vec::with_capacity(k.unwrap_or(intermediate_results.len()));

    for result in intermediate_results {
        let internal_id = InternalId::from(result.vector_id);

        let raw_embedding_ref = collection
            .get_raw_emb_by_internal_id(&internal_id)
            .ok_or_else(|| WaCustomError::NotFound("raw embedding not found".to_string()))?;

        let sparse_pairs = raw_embedding_ref.sparse_values.clone().ok_or_else(|| {
            WaCustomError::NotFound("sparse values is missing in raw embedding".to_string())
        })?;
        let map: std::collections::HashMap<u32, f32> =
            sparse_pairs.iter().map(|sp| (sp.0, sp.1)).collect();

        let mut dp = 0.0;
        for pair in query {
            if let Some(val) = map.get(&pair.0) {
                dp += val * pair.1;
            }
        }

        results.push((
            internal_id,
            Some(raw_embedding_ref.id.clone()),
            raw_embedding_ref.document_id.clone(),
            dp,
            if return_raw_text {
                raw_embedding_ref.text.clone()
            } else {
                None
            },
        ));
    }

    // Sort by descending order
    results.sort_unstable_by(|(_, _, _, a, _), (_, _, _, b, _)| b.total_cmp(a));

    if let Some(k_val) = k {
        results.truncate(k_val);
    }

    Ok(results)
}
