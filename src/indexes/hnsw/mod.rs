pub(crate) mod types;

use std::sync::{
    atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
    Arc, RwLock,
};

use types::HNSWHyperParams;

use crate::{
    config_loader::Config,
    metadata::MetadataFields,
    models::{
        buffered_io::BufferManagerFactory,
        cache_loader::HNSWIndexCache,
        collection::Collection,
        collection_transaction::CollectionTransaction,
        common::WaCustomError,
        meta_persist::store_values_range,
        prob_lazy_load::lazy_item::{FileIndex, ProbLazyItem},
        prob_node::{ProbNode, SharedNode},
        types::{DistanceMetric, MetaDb, QuantizationMetric, VectorId},
        versioning::Hash,
    },
    quantization::StorageType,
    vector_store::index_embeddings_in_transaction,
};

use super::IndexOps;

pub struct DenseInputEmbedding(
    pub VectorId,
    pub Vec<f32>,
    pub Option<MetadataFields>,
    pub bool,
);

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct HNSWIndexData {
    pub hnsw_params: HNSWHyperParams,
    pub levels_prob: Vec<(f64, u8)>,
    pub dim: usize,
    pub file_index: FileIndex,
    pub quantization_metric: QuantizationMetric,
    pub distance_metric: DistanceMetric,
    pub storage_type: StorageType,
    pub sample_threshold: usize,
}

pub struct HNSWIndex {
    pub root_vec: AtomicPtr<ProbLazyItem<ProbNode>>,
    pub levels_prob: Vec<(f64, u8)>,
    pub dim: usize,
    pub quantization_metric: RwLock<QuantizationMetric>,
    pub distance_metric: Arc<RwLock<DistanceMetric>>,
    pub storage_type: RwLock<StorageType>,
    pub hnsw_params: RwLock<HNSWHyperParams>,
    pub cache: HNSWIndexCache,
    pub vec_raw_manager: BufferManagerFactory<Hash>,
    pub is_configured: AtomicBool,
    pub values_range: RwLock<(f32, f32)>,
    pub vectors: RwLock<Vec<DenseInputEmbedding>>,
    pub sampling_data: SamplingData,
    pub vectors_collected: AtomicUsize,
    pub sample_threshold: usize,
}

#[derive(Default)]
pub struct SamplingData {
    pub above_05: AtomicUsize,
    pub above_04: AtomicUsize,
    pub above_03: AtomicUsize,
    pub above_02: AtomicUsize,
    pub above_01: AtomicUsize,

    pub below_05: AtomicUsize,
    pub below_04: AtomicUsize,
    pub below_03: AtomicUsize,
    pub below_02: AtomicUsize,
    pub below_01: AtomicUsize,
}

unsafe impl Send for HNSWIndex {}
unsafe impl Sync for HNSWIndex {}

impl HNSWIndex {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        root_vec: SharedNode,
        levels_prob: Vec<(f64, u8)>,
        dim: usize,
        quantization_metric: QuantizationMetric,
        distance_metric: Arc<RwLock<DistanceMetric>>,
        storage_type: StorageType,
        hnsw_params: HNSWHyperParams,
        cache: HNSWIndexCache,
        vec_raw_manager: BufferManagerFactory<Hash>,
        values_range: (f32, f32),
        sample_threshold: usize,
        is_configured: bool,
    ) -> Self {
        Self {
            root_vec: AtomicPtr::new(root_vec),
            levels_prob,
            dim,
            quantization_metric: RwLock::new(quantization_metric),
            distance_metric,
            storage_type: RwLock::new(storage_type),
            hnsw_params: RwLock::new(hnsw_params),
            cache,
            vec_raw_manager,
            is_configured: AtomicBool::new(is_configured),
            values_range: RwLock::new(values_range),
            vectors: RwLock::new(Vec::new()),
            sampling_data: SamplingData::default(),
            vectors_collected: AtomicUsize::new(0),
            sample_threshold,
        }
    }

    pub fn get_root_vec(&self) -> SharedNode {
        self.root_vec.load(Ordering::SeqCst)
    }

    /// Returns FileIndex (offset) corresponding to the root node.
    pub fn root_vec_offset(&self) -> FileIndex {
        unsafe { &*self.get_root_vec() }.get_file_index()
    }
}

impl IndexOps for HNSWIndex {
    type InputEmbedding = DenseInputEmbedding;
    type Data = HNSWIndexData;

    fn index_embeddings(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::InputEmbedding>,
        transaction: &CollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        index_embeddings_in_transaction(config, collection, self, transaction, embeddings)
    }

    fn sample_embedding(&self, embedding: &Self::InputEmbedding) {
        for value in &embedding.1 {
            let value = *value;

            if value > 0.1 {
                self.sampling_data.above_01.fetch_add(1, Ordering::Relaxed);
            }

            if value > 0.2 {
                self.sampling_data.above_02.fetch_add(1, Ordering::Relaxed);
            }

            if value > 0.3 {
                self.sampling_data.above_03.fetch_add(1, Ordering::Relaxed);
            }

            if value > 0.4 {
                self.sampling_data.above_04.fetch_add(1, Ordering::Relaxed);
            }

            if value > 0.5 {
                self.sampling_data.above_05.fetch_add(1, Ordering::Relaxed);
            }

            if value < -0.1 {
                self.sampling_data.below_01.fetch_add(1, Ordering::Relaxed);
            }

            if value < -0.2 {
                self.sampling_data.below_02.fetch_add(1, Ordering::Relaxed);
            }

            if value < -0.3 {
                self.sampling_data.below_03.fetch_add(1, Ordering::Relaxed);
            }

            if value < -0.4 {
                self.sampling_data.below_04.fetch_add(1, Ordering::Relaxed);
            }

            if value < -0.5 {
                self.sampling_data.below_05.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn finalize_sampling(
        &self,
        lmdb: &MetaDb,
        config: &Config,
        embeddings: &[Self::InputEmbedding],
    ) -> Result<(), WaCustomError> {
        let dimension = embeddings
            .first()
            .map(|embedding| embedding.1.len())
            .unwrap_or_default();
        let values_count = (dimension * embeddings.len()) as f32;

        let above_05_percent =
            (self.sampling_data.above_05.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_04_percent =
            (self.sampling_data.above_04.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_03_percent =
            (self.sampling_data.above_03.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_02_percent =
            (self.sampling_data.above_02.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let above_01_percent =
            (self.sampling_data.above_01.load(Ordering::Relaxed) as f32 / values_count) * 100.0;

        let below_05_percent =
            (self.sampling_data.below_05.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let below_04_percent =
            (self.sampling_data.below_04.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let below_03_percent =
            (self.sampling_data.below_03.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let below_02_percent =
            (self.sampling_data.below_02.load(Ordering::Relaxed) as f32 / values_count) * 100.0;
        let below_01_percent =
            (self.sampling_data.below_01.load(Ordering::Relaxed) as f32 / values_count) * 100.0;

        let range_start = if below_01_percent <= config.indexing.clamp_margin_percent {
            -0.1
        } else if below_02_percent <= config.indexing.clamp_margin_percent {
            -0.2
        } else if below_03_percent <= config.indexing.clamp_margin_percent {
            -0.3
        } else if below_04_percent <= config.indexing.clamp_margin_percent {
            -0.4
        } else if below_05_percent <= config.indexing.clamp_margin_percent {
            -0.5
        } else {
            -1.0
        };

        let range_end = if above_01_percent <= config.indexing.clamp_margin_percent {
            0.1
        } else if above_02_percent <= config.indexing.clamp_margin_percent {
            0.2
        } else if above_03_percent <= config.indexing.clamp_margin_percent {
            0.3
        } else if above_04_percent <= config.indexing.clamp_margin_percent {
            0.4
        } else if above_05_percent <= config.indexing.clamp_margin_percent {
            0.5
        } else {
            1.0
        };

        let range = (range_start, range_end);
        *self.values_range.write().unwrap() = range;
        self.is_configured.store(true, Ordering::Release);
        store_values_range(lmdb, range)?;
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
        self.cache.flush_all()?;
        Ok(())
    }

    fn get_data(&self) -> Self::Data {
        let offset = self.root_vec_offset();
        Self::Data {
            hnsw_params: self.hnsw_params.read().unwrap().clone(),
            levels_prob: self.levels_prob.clone(),
            dim: self.dim,
            file_index: offset,
            quantization_metric: self.quantization_metric.read().unwrap().clone(),
            distance_metric: *self.distance_metric.read().unwrap(),
            storage_type: *self.storage_type.read().unwrap(),
            sample_threshold: self.sample_threshold,
        }
    }
}
