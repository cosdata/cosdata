pub(crate) mod data;
pub(crate) mod transaction;
pub(crate) mod types;

use std::{
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use transaction::HNSWIndexTransaction;
use types::HNSWHyperParams;

use crate::{
    metadata::MetadataFields, models::{
        buffered_io::BufferManagerFactory,
        cache_loader::HNSWIndexCache,
        prob_lazy_load::lazy_item::{FileIndex, ProbLazyItem},
        prob_node::{ProbNode, SharedNode},
        types::{DistanceMetric, MetaDb, QuantizationMetric, VectorId},
        versioning::{Hash, VersionControl},
    }, quantization::StorageType
};

pub struct HNSWIndex {
    pub name: String,
    pub root_vec: AtomicPtr<ProbLazyItem<ProbNode>>,
    pub levels_prob: Vec<(f64, u8)>,
    // @TODO(vineet): Should this be with or without phantom dimensions
    pub dim: usize,
    pub lmdb: MetaDb,
    pub current_version: RwLock<Hash>,
    pub current_open_transaction: AtomicPtr<HNSWIndexTransaction>,
    pub quantization_metric: RwLock<QuantizationMetric>,
    pub distance_metric: Arc<RwLock<DistanceMetric>>,
    pub storage_type: RwLock<StorageType>,
    pub vcs: VersionControl,
    pub hnsw_params: RwLock<HNSWHyperParams>,
    pub cache: HNSWIndexCache,
    pub vec_raw_manager: BufferManagerFactory<Hash>,
    pub is_configured: AtomicBool,
    pub values_range: RwLock<(f32, f32)>,
    pub vectors: RwLock<Vec<(VectorId, Vec<f32>, Option<MetadataFields>)>>,
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
        name: String,
        root_vec: SharedNode,
        levels_prob: Vec<(f64, u8)>,
        dim: usize,
        lmdb: MetaDb,
        current_version: Hash,
        quantization_metric: QuantizationMetric,
        distance_metric: Arc<RwLock<DistanceMetric>>,
        storage_type: StorageType,
        vcs: VersionControl,
        hnsw_params: HNSWHyperParams,
        cache: HNSWIndexCache,
        vec_raw_manager: BufferManagerFactory<Hash>,
        values_range: (f32, f32),
        sample_threshold: usize,
        is_configured: bool,
    ) -> Self {
        Self {
            name,
            root_vec: AtomicPtr::new(root_vec),
            levels_prob,
            dim,
            lmdb,
            current_version: RwLock::new(current_version),
            current_open_transaction: AtomicPtr::new(ptr::null_mut()),
            quantization_metric: RwLock::new(quantization_metric),
            distance_metric,
            storage_type: RwLock::new(storage_type),
            vcs,
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

    // Get method
    pub fn get_current_version(&self) -> Hash {
        *self.current_version.read().unwrap()
    }

    // Set method
    pub fn set_current_version(&self, new_version: Hash) {
        *self.current_version.write().unwrap() = new_version;
    }

    pub fn get_root_vec(&self) -> SharedNode {
        self.root_vec.load(Ordering::SeqCst)
    }

    /// Returns FileIndex (offset) corresponding to the root node.
    pub fn root_vec_offset(&self) -> FileIndex {
        unsafe { &*self.get_root_vec() }.get_file_index()
    }
}
