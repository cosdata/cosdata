pub(crate) mod types;
use crate::{
    config_loader::Config,
    models::{
        collection::Collection, collection_transaction::CollectionTransaction,
        common::WaCustomError, meta_persist::store_values_upper_bound, types::MetaDb,
    },
};

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use types::{RawSparseVectorEmbedding, SamplingData, SparsePair};

use crate::models::{
    buffered_io::{BufIoError, BufferManagerFactory},
    inverted_index::InvertedIndexRoot,
    tree_map::TreeMap,
    types::VectorId,
    versioning::Hash,
};

use super::IndexOps;

pub struct SparseInputEmbedding(pub VectorId, pub Vec<SparsePair>);

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
    pub vec_raw_manager: BufferManagerFactory<u8>,
    pub vec_raw_map: TreeMap<RawSparseVectorEmbedding>,
}

unsafe impl Send for InvertedIndex {}
unsafe impl Sync for InvertedIndex {}

impl InvertedIndex {
    pub fn new(
        root_path: PathBuf,
        vec_raw_manager: BufferManagerFactory<u8>,
        quantization_bits: u8,
        sample_threshold: usize,
        data_file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let root = InvertedIndexRoot::new(root_path, quantization_bits, data_file_parts)?;

        Ok(Self {
            root,
            values_upper_bound: RwLock::new(1.0),
            is_configured: AtomicBool::new(false),
            sampling_data: SamplingData::default(),
            vectors: RwLock::new(Vec::new()),
            vectors_collected: AtomicUsize::new(0),
            sample_threshold,
            vec_raw_manager,
            vec_raw_map: TreeMap::new(),
        })
    }

    pub fn insert(
        &self,
        id: VectorId,
        pairs: Vec<SparsePair>,
        version: Hash,
    ) -> Result<(), BufIoError> {
        for pair in &pairs {
            self.root.insert(
                pair.0,
                pair.1,
                id.0 as u32,
                version,
                *self.values_upper_bound.read().unwrap(),
            )?;
        }

        let emb = RawSparseVectorEmbedding {
            raw_vec: Arc::new(pairs),
            hash_vec: id.clone(),
        };

        self.vec_raw_map.insert(version, id.0, emb);
        Ok(())
    }
}

impl IndexOps for InvertedIndex {
    type InputEmbedding = SparseInputEmbedding;
    type Data = InvertedIndexData;

    fn index_embeddings(
        &self,
        _collection: &Collection,
        embeddings: Vec<Self::InputEmbedding>,
        transaction: &CollectionTransaction,
        _config: &Config,
    ) -> Result<(), WaCustomError> {
        embeddings
            .into_par_iter()
            .try_for_each(|SparseInputEmbedding(id, pairs)| {
                self.insert(id, pairs, transaction.id)
            })?;
        Ok(())
    }

    fn sample_embedding(&self, embedding: &Self::InputEmbedding) {
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
        _embeddings: &[Self::InputEmbedding],
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

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::InputEmbedding>> {
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

    fn flush(&self, _: &Collection) -> Result<(), WaCustomError> {
        self.vec_raw_map
            .serialize(&self.vec_raw_manager, self.root.data_file_parts)?;
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
}
