pub(crate) mod offset_counter;
pub(crate) mod types;

use super::{IndexOps, InternalSearchResult};
use crate::{
    config_loader::Config,
    metadata::{
        query_filtering::{filter_encoded_dimensions, Filter},
        MetadataFields,
    },
    models::{
        cache_loader::HNSWIndexCache,
        collection::{Collection, RawVectorEmbedding},
        common::{TSHashTable, WaCustomError},
        meta_persist::store_values_range,
        prob_node::SharedLatestNode,
        types::{DistanceMetric, FileOffset, HNSWLevel, InternalId, MetaDb, QuantizationMetric},
        versioning::VersionNumber,
    },
    quantization::{Quantization, StorageType},
    vector_store::{ann_search, delete_embedding, finalize_ann_results, index_embeddings},
};
use offset_counter::HNSWIndexFileOffsetCounter;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, RwLock,
};
use types::{HNSWHyperParams, QuantizedDenseVectorEmbedding};

pub struct DenseInputEmbedding(
    pub InternalId,
    /// Raw vector embedding
    pub Vec<f32>,
    /// Optional metadata fields
    pub Option<MetadataFields>,
    /// Boolean flag to indicate pseudo nodes
    pub bool,
);

pub struct DenseSearchInput(pub Vec<f32>, pub Option<Filter>);

pub struct DenseSearchOptions {
    pub top_k: Option<usize>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct HNSWIndexData {
    pub hnsw_params: HNSWHyperParams,
    pub levels_prob: Vec<(f64, u8)>,
    pub dim: usize,
    pub root_vec_ptr_offset: FileOffset,
    pub pseudo_root_vec_ptr_offset: Option<FileOffset>,
    pub quantization_metric: QuantizationMetric,
    pub distance_metric: DistanceMetric,
    pub storage_type: StorageType,
    pub sample_threshold: usize,
}

pub struct HNSWIndex {
    pub root_vec: SharedLatestNode,
    pub pseudo_root_vec: Option<SharedLatestNode>,
    pub levels_prob: Vec<(f64, u8)>,
    pub dim: usize,
    pub quantization_metric: RwLock<QuantizationMetric>,
    pub distance_metric: Arc<RwLock<DistanceMetric>>,
    pub storage_type: RwLock<StorageType>,
    pub hnsw_params: RwLock<HNSWHyperParams>,
    pub cache: HNSWIndexCache,
    pub is_configured: AtomicBool,
    pub values_range: RwLock<(f32, f32)>,
    pub vectors: RwLock<Vec<DenseInputEmbedding>>,
    pub sampling_data: SamplingData,
    pub vectors_collected: AtomicUsize,
    pub sample_threshold: usize,
    pub max_replica_per_node: u8,
    pub offset_counter: RwLock<HNSWIndexFileOffsetCounter>,
    pub versions_synchronization_map: TSHashTable<SharedLatestNode, ()>,
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
        root_vec: SharedLatestNode,
        pseudo_root_vec: Option<SharedLatestNode>,
        levels_prob: Vec<(f64, u8)>,
        dim: usize,
        quantization_metric: QuantizationMetric,
        distance_metric: Arc<RwLock<DistanceMetric>>,
        storage_type: StorageType,
        hnsw_params: HNSWHyperParams,
        cache: HNSWIndexCache,
        values_range: (f32, f32),
        sample_threshold: usize,
        is_configured: bool,
        max_replica_per_node: u8,
        offset_counter: HNSWIndexFileOffsetCounter,
    ) -> Self {
        Self {
            root_vec,
            pseudo_root_vec,
            levels_prob,
            dim,
            quantization_metric: RwLock::new(quantization_metric),
            distance_metric,
            storage_type: RwLock::new(storage_type),
            hnsw_params: RwLock::new(hnsw_params),
            cache,
            is_configured: AtomicBool::new(is_configured),
            values_range: RwLock::new(values_range),
            vectors: RwLock::new(Vec::new()),
            sampling_data: SamplingData::default(),
            vectors_collected: AtomicUsize::new(0),
            sample_threshold,
            max_replica_per_node,
            offset_counter: RwLock::new(offset_counter),
            versions_synchronization_map: TSHashTable::new(16),
        }
    }

    pub fn get_root_vec(&self) -> SharedLatestNode {
        self.root_vec
    }

    pub fn get_pseudo_root_vec(&self) -> Option<SharedLatestNode> {
        self.pseudo_root_vec
    }

    /// Returns FileIndex (offset) corresponding to the root node.
    pub fn root_vec_ptr_offset(&self) -> FileOffset {
        unsafe { &*self.root_vec }.file_offset
    }

    /// Returns FileIndex (offset) corresponding to the pseudo root node.
    pub fn pseudo_root_vec_ptr_offset(&self) -> Option<FileOffset> {
        let node = unsafe { self.get_pseudo_root_vec().map(|node| &*node) };
        node.map(|n| n.file_offset)
    }
}

impl IndexOps for HNSWIndex {
    type IndexingInput = DenseInputEmbedding;
    type SearchInput = DenseSearchInput;
    type SearchOptions = DenseSearchOptions;
    type Data = HNSWIndexData;

    fn validate_embedding(&self, embedding: Self::IndexingInput) -> Result<(), WaCustomError> {
        // @TODO(vineet): Add validation for metadata fields (if
        // applicable)
        if embedding.1.len() == self.dim {
            Ok(())
        } else {
            Err(WaCustomError::InvalidData(format!(
                "Expected dimension of dense vector to be {}, found {}",
                self.dim,
                embedding.1.len()
            )))
        }
    }

    fn index_embeddings(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::IndexingInput>,
        version: VersionNumber,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        index_embeddings(config, collection, self, version, embeddings)
    }

    fn delete_embedding(
        &self,
        id: InternalId,
        raw_emb: &RawVectorEmbedding,
        version: VersionNumber,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        delete_embedding(config, self, version, id, raw_emb)
    }

    fn sample_embedding(&self, embedding: &Self::IndexingInput) {
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
        embeddings: &[Self::IndexingInput],
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

    fn flush(&self, _: &Collection, version: VersionNumber) -> Result<(), WaCustomError> {
        self.cache.flush_all(version)?;
        Ok(())
    }

    fn get_data(&self) -> Self::Data {
        let offset = self.root_vec_ptr_offset();
        let offset_pseudo = self.pseudo_root_vec_ptr_offset();
        Self::Data {
            hnsw_params: self.hnsw_params.read().unwrap().clone(),
            levels_prob: self.levels_prob.clone(),
            dim: self.dim,
            root_vec_ptr_offset: offset,
            pseudo_root_vec_ptr_offset: offset_pseudo,
            quantization_metric: self.quantization_metric.read().unwrap().clone(),
            distance_metric: *self.distance_metric.read().unwrap(),
            storage_type: *self.storage_type.read().unwrap(),
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
        let id = InternalId::from(u32::MAX - 1);
        let quantized_vec = self.quantization_metric.read().unwrap().quantize(
            &query.0,
            *self.storage_type.read().unwrap(),
            *self.values_range.read().unwrap(),
        )?;
        let vec_emb = QuantizedDenseVectorEmbedding {
            quantized_vec: Arc::new(quantized_vec),
            hash_vec: id,
        };

        let hnsw_params_guard = self.hnsw_params.read().unwrap();

        let query_filter_dims = query.1.as_ref().map(|filter| {
            let metadata_schema = collection.meta.metadata_schema.as_ref().unwrap();
            filter_encoded_dimensions(metadata_schema, filter).unwrap()
        });

        let root_node = if query.1.is_some() {
            self.get_pseudo_root_vec().unwrap()
        } else {
            self.get_root_vec()
        };

        let results = ann_search(
            config,
            self,
            vec_emb,
            query_filter_dims.as_ref(),
            root_node,
            HNSWLevel(hnsw_params_guard.num_layers),
            &hnsw_params_guard,
        )?;
        drop(hnsw_params_guard);
        finalize_ann_results(
            collection,
            self,
            results,
            &query.0,
            options.top_k,
            return_raw_text,
        )
    }
}
