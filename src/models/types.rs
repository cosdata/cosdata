use super::buffered_io::BufferManagerFactory;
use super::cache_loader::ProbCache;
use super::collection::Collection;
use super::crypto::{DoubleSHA256Hash, SingleSHA256Hash};
use super::embedding_persist::{write_embedding, EmbeddingOffset};
use super::meta_persist::{
    delete_dense_index, lmdb_init_collections_db, lmdb_init_db, load_collections,
    load_dense_index_data, persist_dense_index, retrieve_current_version,
};
use super::prob_lazy_load::lazy_item::ProbLazyItem;
use super::prob_node::{ProbNode, SharedNode};
use super::versioning::VersionControl;
use crate::config_loader::Config;
use crate::distance::cosine::CosineSimilarity;
use crate::distance::DistanceError;
use crate::distance::{
    cosine::CosineDistance, dotproduct::DotProductDistance, euclidean::EuclideanDistance,
    hamming::HammingDistance, DistanceFunction,
};
use crate::indexes::inverted_index::InvertedIndex;
use crate::indexes::inverted_index_data::InvertedIndexData;
use crate::macros::key;
use crate::models::buffered_io::BufIoError;
use crate::models::common::*;
use crate::models::identity_collections::*;
use crate::models::lazy_load::*;
use crate::models::meta_persist::retrieve_values_range;
use crate::models::versioning::*;
use crate::quantization::{
    product::ProductQuantization, scalar::ScalarQuantization, Quantization, QuantizationError,
    StorageType,
};
use crate::storage::Storage;
use arcshift::ArcShift;
use dashmap::DashMap;
use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rpassword::prompt_password;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher24;
use std::hash::{DefaultHasher, Hash as StdHash, Hasher};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::time::Instant;
use std::{fmt, ptr};
use std::{fs::*, thread};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HNSWLevel(pub u8);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileOffset(pub u32);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BytesToRead(pub u32);

#[derive(Clone)]
pub struct Neighbour {
    pub node: LazyItem<MergedNode>,
    pub cosine_similarity: CosineSimilarity,
}

impl Identifiable for Neighbour {
    type Id = LazyItemId;

    fn get_id(&self) -> Self::Id {
        self.node.get_id()
    }
}

impl Identifiable for MergedNode {
    type Id = u64;

    fn get_id(&self) -> Self::Id {
        let mut prop_ref = self.prop.clone();
        let prop = prop_ref.get();
        let mut hasher = DefaultHasher::new();
        prop.hash(&mut hasher);
        hasher.finish()
    }
}

pub type PropPersistRef = (FileOffset, BytesToRead);

#[derive(Debug, Clone, PartialEq)]
pub struct NodeProp {
    pub id: VectorId,
    pub value: Arc<Storage>,
    pub location: PropPersistRef,
}

impl StdHash for NodeProp {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Hash)]
pub enum PropState {
    Ready(Arc<NodeProp>),
    Pending(PropPersistRef),
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct VectorId(pub u64);

impl VectorId {
    pub fn get_hash(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone)]
pub struct MergedNode {
    pub hnsw_level: HNSWLevel,
    pub prop: ArcShift<PropState>,
    pub neighbors: EagerLazyItemSet<MergedNode, MetricResult>,
    pub parent: LazyItemRef<MergedNode>,
    pub child: LazyItemRef<MergedNode>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, PartialOrd)]
pub enum MetricResult {
    CosineSimilarity(CosineSimilarity),
    CosineDistance(CosineDistance),
    EuclideanDistance(EuclideanDistance),
    HammingDistance(HammingDistance),
    DotProductDistance(DotProductDistance),
}

impl Eq for MetricResult {}

impl Ord for MetricResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl MetricResult {
    // gets the bare numerical value stored in the type
    pub fn get_value(&self) -> f32 {
        match self {
            MetricResult::CosineSimilarity(value) => value.0,
            MetricResult::CosineDistance(value) => value.0,
            MetricResult::EuclideanDistance(value) => value.0,
            MetricResult::HammingDistance(value) => value.0,
            MetricResult::DotProductDistance(value) => value.0,
        }
    }

    pub fn get_tag_and_value(&self) -> (u8, f32) {
        match self {
            Self::CosineSimilarity(value) => (0, value.0),
            Self::CosineDistance(value) => (1, value.0),
            Self::EuclideanDistance(value) => (2, value.0),
            Self::HammingDistance(value) => (3, value.0),
            Self::DotProductDistance(value) => (4, value.0),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Hamming,
    DotProduct,
}

impl DistanceFunction for DistanceMetric {
    type Item = MetricResult;
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<Self::Item, DistanceError> {
        match self {
            Self::Cosine => {
                let value = CosineSimilarity(0.0).calculate(x, y)?;
                Ok(MetricResult::CosineSimilarity(value))
            }
            Self::Euclidean => {
                let value = EuclideanDistance(0.0).calculate(x, y)?;
                Ok(MetricResult::EuclideanDistance(value))
            }
            Self::Hamming => {
                let value = HammingDistance(0.0).calculate(x, y)?;
                Ok(MetricResult::HammingDistance(value))
            }
            Self::DotProduct => {
                let value = DotProductDistance(0.0).calculate(x, y)?;
                Ok(MetricResult::DotProductDistance(value))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMetric {
    Scalar,
    Product(ProductQuantization),
}

impl Quantization for QuantizationMetric {
    fn quantize(
        &self,
        vector: &[f32],
        storage_type: StorageType,
        range: (f32, f32),
    ) -> Result<Storage, QuantizationError> {
        match self {
            Self::Scalar => ScalarQuantization.quantize(vector, storage_type, range),
            Self::Product(product) => product.quantize(vector, storage_type, range),
        }
    }

    fn train(&mut self, vectors: &[&[f32]]) -> Result<(), QuantizationError> {
        match self {
            Self::Scalar => ScalarQuantization.train(vectors),
            Self::Product(product) => product.train(vectors),
        }
    }
}

impl MergedNode {
    pub fn new(hnsw_level: HNSWLevel) -> Self {
        MergedNode {
            hnsw_level,
            prop: ArcShift::new(PropState::Pending((FileOffset(0), BytesToRead(0)))),
            neighbors: EagerLazyItemSet::new(),
            parent: LazyItemRef::new_invalid(),
            child: LazyItemRef::new_invalid(),
        }
    }

    pub fn add_ready_neighbor(&self, neighbor: LazyItem<MergedNode>, distance: MetricResult) {
        self.neighbors.insert(EagerLazyItem(distance, neighbor));
    }

    pub fn set_parent(&self, parent: LazyItem<MergedNode>) {
        let mut arc = self.parent.item.clone();
        arc.update(parent);
    }

    pub fn set_child(&self, child: LazyItem<MergedNode>) {
        let mut arc = self.child.item.clone();
        arc.update(child);
    }

    pub fn add_ready_neighbors(&self, neighbors_list: Vec<(LazyItem<MergedNode>, MetricResult)>) {
        for (neighbor, distance) in neighbors_list {
            self.add_ready_neighbor(neighbor, distance);
        }
    }

    pub fn get_neighbors(&self) -> EagerLazyItemSet<MergedNode, MetricResult> {
        self.neighbors.clone()
    }

    pub fn get_parent(&self) -> LazyItemRef<MergedNode> {
        self.parent.clone()
    }

    pub fn get_child(&self) -> LazyItemRef<MergedNode> {
        self.child.clone()
    }

    pub fn get_prop_location(&self) -> PropPersistRef {
        let mut arc = self.prop.clone();
        match arc.get() {
            PropState::Ready(node_prop) => node_prop.location.clone(),
            PropState::Pending(location) => *location,
        }
    }

    pub fn get_prop(&self) -> PropState {
        let mut arc = self.prop.clone();
        arc.get().clone()
    }

    pub fn set_prop_pending(&self, prop_ref: PropPersistRef) {
        let mut arc = self.prop.clone();
        arc.update(PropState::Pending(prop_ref));
    }

    pub fn set_prop_ready(&self, node_prop: Arc<NodeProp>) {
        let mut arc = self.prop.clone();
        arc.update(PropState::Ready(node_prop));
    }
}

// Implementing the std::fmt::Display trait for VectorId
impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for MergedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MergedNode {{")?;
        // writeln!(f, "  version_id: {},", self.version_id.0)?;
        writeln!(f, "  hnsw_level: {},", self.hnsw_level.0)?;

        // Display PropState
        write!(f, "  prop: ")?;
        let mut prop_arc = self.prop.clone();
        let prop = match prop_arc.get() {
            PropState::Ready(node_prop) => format!("Ready {{ id: {} }}", node_prop.id),
            PropState::Pending(_) => "Pending".to_string(),
        };
        f.debug_struct("MergedNode")
            .field("hnsw_level", &self.hnsw_level)
            .field("prop", &prop)
            .field("neighbors", &self.neighbors.len())
            .field(
                "parent",
                if self.parent.is_valid() {
                    &"Valid"
                } else {
                    &"Invalid"
                },
            )
            .field(
                "child",
                if self.child.is_valid() {
                    &"Valid"
                } else {
                    &"Invalid"
                },
            )
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorQt {
    UnsignedByte {
        mag: u32,
        quant_vec: Vec<u8>,
    },
    SubByte {
        mag: u32,
        quant_vec: Vec<Vec<u8>>,
        resolution: u8,
    },
}

impl VectorQt {
    pub fn unsigned_byte(vec: &[f32]) -> Self {
        let quant_vec = simp_quant(vec)
            .inspect_err(|x| println!("{:?}", x))
            .unwrap();
        let mag = mag_square_u8(&quant_vec);
        Self::UnsignedByte { mag, quant_vec }
    }

    pub fn sub_byte(vec: &[f32], resolution: u8) -> Self {
        let quant_vec = quantize_to_u8_bits(vec, resolution);
        let mag = 0; //implement a proper magnitude calculation
        Self::SubByte {
            mag,
            quant_vec,
            resolution,
        }
    }
}

// #[allow(dead_code)]
// pub struct SizeBytes(pub u32);

#[derive(Debug, Clone)]
pub struct MetaDb {
    pub env: Arc<Environment>,
    pub db: Arc<Database>,
}

impl MetaDb {
    pub fn from_env(env: Arc<Environment>, collection_name: &str) -> lmdb::Result<Self> {
        let db = Arc::new(env.create_db(Some(collection_name), DatabaseFlags::empty())?);

        Ok(Self { env, db })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWHyperParams {
    pub num_layers: u8,
    pub ef_construction: u32,
    pub ef_search: u32,
    pub max_cache_size: usize,
    pub level_0_neighbors_count: usize,
    pub neighbors_count: usize,
}

impl HNSWHyperParams {
    pub fn default_from_config(config: &Config) -> Self {
        Self {
            ef_construction: config.hnsw.default_ef_construction,
            ef_search: config.hnsw.default_ef_search,
            num_layers: config.hnsw.default_num_layer,
            max_cache_size: config.hnsw.default_max_cache_size,
            level_0_neighbors_count: config.hnsw.default_level_0_neighbors_count,
            neighbors_count: config.hnsw.default_neighbors_count,
        }
    }
}

pub struct DenseIndexTransaction {
    pub id: Hash,
    pub version_number: u16,
    pub lazy_item_versions_table: Arc<TSHashTable<(VectorId, u16, u8), SharedNode>>,
    raw_embedding_serializer_thread_handle: thread::JoinHandle<Result<(), WaCustomError>>,
    pub raw_embedding_channel: mpsc::Sender<RawVectorEmbedding>,
    level_0_node_offset_counter: AtomicU32,
    node_offset_counter: AtomicU32,
    node_size: u32,
    level_0_node_size: u32,
}

unsafe impl Send for DenseIndexTransaction {}
unsafe impl Sync for DenseIndexTransaction {}

impl DenseIndexTransaction {
    pub fn new(dense_index: Arc<DenseIndex>) -> Result<Self, WaCustomError> {
        let branch_info = dense_index
            .vcs
            .get_branch_info("main")
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get main branch info: {}", err))
            })?
            .unwrap();
        let version_number = *branch_info.get_current_version() + 1;
        let id = dense_index
            .vcs
            .generate_hash("main", Version::from(version_number))
            .map_err(|err| {
                WaCustomError::DatabaseError(format!("Unable to get transaction hash: {}", err))
            })?;

        let (raw_embedding_channel, rx) = mpsc::channel();

        let raw_embedding_serializer_thread_handle = {
            let bufman = dense_index.vec_raw_manager.get(id)?;
            let dense_index = dense_index.clone();

            thread::spawn(move || {
                let mut offsets = Vec::new();
                for raw_emb in rx {
                    let offset = write_embedding(bufman.clone(), &raw_emb)?;
                    let embedding_key = key!(e:raw_emb.hash_vec);
                    offsets.push((embedding_key, offset));
                }

                let env = dense_index.lmdb.env.clone();
                let db = dense_index.lmdb.db.clone();

                let mut txn = env.begin_rw_txn().map_err(|e| {
                    WaCustomError::DatabaseError(format!("Failed to begin transaction: {}", e))
                })?;
                for (key, offset) in offsets {
                    let offset = EmbeddingOffset {
                        version: id,
                        offset,
                    };
                    let offset_serialized = offset.serialize();

                    txn.put(*db, &key, &offset_serialized, WriteFlags::empty())
                        .map_err(|e| {
                            WaCustomError::DatabaseError(format!("Failed to put data: {}", e))
                        })?;
                }

                txn.commit().map_err(|e| {
                    WaCustomError::DatabaseError(format!("Failed to commit transaction: {}", e))
                })?;
                let start = Instant::now();
                bufman.flush()?;
                println!("Time took to flush: {:?}", start.elapsed());
                Ok(())
            })
        };

        let hnsw_params = dense_index.hnsw_params.read().unwrap();

        Ok(Self {
            id,
            lazy_item_versions_table: Arc::new(TSHashTable::new(16)),
            raw_embedding_channel,
            raw_embedding_serializer_thread_handle,
            version_number: version_number as u16,
            node_offset_counter: AtomicU32::new(0),
            level_0_node_offset_counter: AtomicU32::new(0),
            node_size: ProbNode::get_serialized_size(hnsw_params.neighbors_count) as u32,
            level_0_node_size: ProbNode::get_serialized_size(hnsw_params.level_0_neighbors_count)
                as u32,
        })
    }

    pub fn post_raw_embedding(&self, raw_emb: RawVectorEmbedding) {
        self.raw_embedding_channel.send(raw_emb).unwrap();
    }

    pub fn pre_commit(self, dense_index: Arc<DenseIndex>) -> Result<(), WaCustomError> {
        dense_index.index_manager.flush_all()?;
        dense_index.level_0_index_manager.flush_all()?;
        dense_index.prop_file.write().unwrap().flush().unwrap();
        drop(self.raw_embedding_channel);
        let start = Instant::now();
        self.raw_embedding_serializer_thread_handle
            .join()
            .unwrap()?;
        println!(
            "Time took to wait for embedding serializer thread: {:?}",
            start.elapsed()
        );
        Ok(())
    }

    pub fn get_new_node_offset(&self) -> u32 {
        self.node_offset_counter
            .fetch_add(self.node_size, Ordering::SeqCst)
    }

    pub fn get_new_level_0_node_offset(&self) -> u32 {
        self.level_0_node_offset_counter
            .fetch_add(self.level_0_node_size, Ordering::SeqCst)
    }
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

#[derive(Clone)]
pub struct DenseIndex {
    pub database_name: String,
    pub root_vec: Arc<AtomicPtr<ProbLazyItem<ProbNode>>>,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub dim: usize,
    pub prop_file: Arc<RwLock<File>>,
    pub lmdb: MetaDb,
    pub current_version: ArcShift<Hash>,
    pub current_open_transaction: Arc<AtomicPtr<DenseIndexTransaction>>,
    pub quantization_metric: ArcShift<QuantizationMetric>,
    pub distance_metric: ArcShift<DistanceMetric>,
    pub storage_type: ArcShift<StorageType>,
    pub vcs: Arc<VersionControl>,
    pub hnsw_params: Arc<RwLock<HNSWHyperParams>>,
    pub cache: Arc<ProbCache>,
    pub index_manager: Arc<BufferManagerFactory<Hash>>,
    pub level_0_index_manager: Arc<BufferManagerFactory<Hash>>,
    pub vec_raw_manager: Arc<BufferManagerFactory<Hash>>,
    pub is_configured: Arc<AtomicBool>,
    pub values_range: Arc<RwLock<(f32, f32)>>,
    pub vectors: Arc<RwLock<Vec<(VectorId, Vec<f32>)>>>,
    pub sampling_data: Arc<SamplingData>,
    pub vectors_collected: Arc<AtomicUsize>,
    pub sample_threshold: usize,
}

unsafe impl Send for DenseIndex {}
unsafe impl Sync for DenseIndex {}

impl DenseIndex {
    pub fn new(
        database_name: String,
        root_vec: SharedNode,
        levels_prob: Arc<Vec<(f64, i32)>>,
        dim: usize,
        prop_file: Arc<RwLock<File>>,
        lmdb: MetaDb,
        current_version: ArcShift<Hash>,
        quantization_metric: ArcShift<QuantizationMetric>,
        distance_metric: ArcShift<DistanceMetric>,
        storage_type: ArcShift<StorageType>,
        vcs: Arc<VersionControl>,
        hnsw_params: HNSWHyperParams,
        cache: Arc<ProbCache>,
        index_manager: Arc<BufferManagerFactory<Hash>>,
        level_0_index_manager: Arc<BufferManagerFactory<Hash>>,
        vec_raw_manager: Arc<BufferManagerFactory<Hash>>,
        values_range: (f32, f32),
        sample_threshold: usize,
        is_configured: bool,
    ) -> Self {
        DenseIndex {
            database_name,
            root_vec: Arc::new(AtomicPtr::new(root_vec)),
            levels_prob,
            dim,
            prop_file,
            lmdb,
            current_version,
            current_open_transaction: Arc::new(AtomicPtr::new(ptr::null_mut())),
            quantization_metric,
            distance_metric,
            storage_type,
            vcs,
            hnsw_params: Arc::new(RwLock::new(hnsw_params)),
            cache,
            index_manager,
            level_0_index_manager,
            vec_raw_manager,
            is_configured: Arc::new(AtomicBool::new(is_configured)),
            values_range: Arc::new(RwLock::new(values_range)),
            vectors: Arc::new(RwLock::new(Vec::new())),
            sampling_data: Arc::new(SamplingData::default()),
            vectors_collected: Arc::new(AtomicUsize::new(0)),
            sample_threshold,
        }
    }

    // Get method
    pub fn get_current_version(&self) -> Hash {
        let mut arc = self.current_version.clone();
        arc.get().clone()
    }

    // Set method
    pub fn set_current_version(&self, new_version: Hash) {
        let mut arc = self.current_version.clone();
        arc.update(new_version);
    }

    pub fn set_root_vec(&self, root_vec: SharedNode) {
        self.root_vec.store(root_vec, Ordering::SeqCst);
    }

    pub fn get_root_vec(&self) -> SharedNode {
        self.root_vec.load(Ordering::SeqCst)
    }

    /// Returns FileIndex (offset) corresponding to the root
    /// node. Returns None if the it's not set or the root node is an
    /// invalid LazyItem
    pub fn root_vec_offset(&self) -> FileIndex {
        unsafe { &*self.get_root_vec() }.get_file_index()
    }
}

// Quantized vector embedding
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVectorEmbedding {
    pub quantized_vec: Arc<Storage>,
    pub hash_vec: VectorId,
}

// Raw vector embedding
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, PartialEq)]
pub struct RawVectorEmbedding {
    pub raw_vec: Arc<Vec<f32>>,
    pub hash_vec: VectorId,
}

pub struct CollectionsMap {
    /// holds an in-memory map of all dense indexes for all collections
    inner: DashMap<String, Arc<DenseIndex>>,
    inner_inverted_index: DashMap<String, Arc<InvertedIndex>>,
    inner_collections: DashMap<String, Arc<Collection>>,
    lmdb_env: Arc<Environment>,
    // made it public temporarily
    // just to be able to persist collections from outside CollectionsMap
    pub(crate) lmdb_collections_db: Database,
    lmdb_dense_index_db: Database,
    #[allow(dead_code)]
    lmdb_inverted_index_db: Database,
}

impl CollectionsMap {
    fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let collections_db = lmdb_init_collections_db(&env)?;
        let dense_index_db = lmdb_init_db(&env, "dense_indexes")?;
        let inverted_index_db = lmdb_init_db(&env, "inverted_indexes")?;
        let res = Self {
            inner: DashMap::new(),
            inner_inverted_index: DashMap::new(),
            inner_collections: DashMap::new(),
            lmdb_env: env,
            lmdb_collections_db: collections_db,
            lmdb_dense_index_db: dense_index_db,
            lmdb_inverted_index_db: inverted_index_db,
        };
        Ok(res)
    }

    /// Loads collections map from lmdb
    ///
    /// In doing so, the root vec for all collections' dense indexes are loaded into
    /// memory, which also ends up warming the cache (NodeRegistry)
    fn load(env: Arc<Environment>, config: &Config) -> Result<Self, WaCustomError> {
        let collections_map =
            Self::new(env.clone()).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collections = load_collections(
            &collections_map.lmdb_env,
            collections_map.lmdb_collections_db.clone(),
        )
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let root_path = Path::new("./collections");

        // let bufmans = cache.get_bufmans();

        for coll in collections {
            let coll = Arc::new(coll);
            collections_map
                .inner_collections
                .insert(coll.name.clone(), coll.clone());

            // if collection has dense index load it from the lmdb
            if coll.dense_vector.enabled {
                let dense_index = collections_map.load_dense_index(&coll, root_path, config)?;
                collections_map
                    .inner
                    .insert(coll.name.clone(), Arc::new(dense_index));
            }

            // if collection has inverted index load it from the lmdb
            if coll.sparse_vector.enabled {
                let inverted_index =
                    collections_map.load_inverted_index(&coll, root_path, config)?;
                collections_map
                    .inner_inverted_index
                    .insert(coll.name.clone(), Arc::new(inverted_index));
            }
        }
        Ok(collections_map)
    }

    /// loads and initiates the dense index of a collection from lmdb
    ///
    /// In doing so, the root vec for all collections' dense indexes are loaded into
    /// memory, which also ends up warming the cache (NodeRegistry)
    fn load_dense_index(
        &self,
        coll: &Collection,
        root_path: &Path,
        config: &Config,
    ) -> Result<DenseIndex, WaCustomError> {
        let collection_path: Arc<Path> = root_path.join(&coll.name).into();
        let index_path = collection_path.join("dense_hnsw");

        let dense_index_data =
            load_dense_index_data(&self.lmdb_env, self.lmdb_dense_index_db, &coll.get_key())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
        let prop_file = Arc::new(RwLock::new(
            OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(index_path.join("prop.data"))
                .unwrap(),
        ));

        let node_size = ProbNode::get_serialized_size(dense_index_data.hnsw_params.neighbors_count);
        let level_0_node_size =
            ProbNode::get_serialized_size(dense_index_data.hnsw_params.level_0_neighbors_count);

        let bufman_size = node_size * 1000;
        let level_0_bufman_size = level_0_node_size * 1000;

        let index_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            bufman_size,
        ));
        let level_0_index_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}_0.index", **ver)),
            level_0_bufman_size,
        ));
        let vec_raw_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
            8192,
        ));
        let cache = Arc::new(ProbCache::new(
            index_manager.clone(),
            level_0_index_manager.clone(),
            prop_file.clone(),
        ));

        let db = Arc::new(
            self.lmdb_env
                .create_db(Some(&coll.name), DatabaseFlags::empty())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
        );

        let (root_offset, root_version_number, root_version_id) = match dense_index_data.file_index
        {
            FileIndex::Valid {
                offset,
                version_number,
                version_id,
            } => (offset, version_number, version_id),
            FileIndex::Invalid => unreachable!(),
        };

        let root_node_region_offset = root_offset.0 - (root_offset.0 % bufman_size as u32);
        let load_start = Instant::now();
        let region = cache.load_region(
            root_node_region_offset,
            root_version_number,
            root_version_id,
            node_size as u32,
            false,
        )?;

        let root = region[(root_offset.0 - root_node_region_offset) as usize / node_size];

        let vcs = Arc::new(VersionControl::from_existing(
            self.lmdb_env.clone(),
            db.clone(),
        ));

        let mut versions = vcs
            .get_branch_versions("main")
            .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

        println!(
            "versions: {:?}",
            versions.iter().map(|(hash, _)| **hash).collect::<Vec<_>>()
        );

        // root node region is already loaded
        let mut num_regions_queued = 1;
        let mut regions_to_load = Vec::new();

        while !versions.is_empty() {
            let num_regions_to_load =
                ((config.num_regions_to_load_on_restart - num_regions_queued + versions.len() - 1)
                    / versions.len())
                    / 2;
            let (version_id, version_hash) = versions.remove(0);
            // level n
            let bufman = index_manager.get(version_id)?;
            for i in 0..num_regions_to_load
                .min((bufman.file_size() as usize + bufman_size - 1) / bufman_size)
            {
                let region_start = (bufman_size * i) as u32;
                if version_id == root_version_id && region_start == root_node_region_offset {
                    continue;
                }
                regions_to_load.push((
                    region_start,
                    *version_hash.version as u16,
                    version_id,
                    node_size as u32,
                    false,
                ));
                num_regions_queued += 1;
            }
            // level 0
            let bufman = level_0_index_manager.get(version_id)?;
            for i in 0..num_regions_to_load
                .min((bufman.file_size() as usize + level_0_bufman_size - 1) / level_0_bufman_size)
            {
                let region_start = (level_0_bufman_size * i) as u32;
                regions_to_load.push((
                    region_start,
                    *version_hash.version as u16,
                    version_id,
                    level_0_node_size as u32,
                    true,
                ));
                num_regions_queued += 1;
            }
        }

        regions_to_load
            .into_par_iter()
            .map(
                |(region_start, version_number, version_id, node_size, is_level_0)| {
                    cache.load_region(
                        region_start,
                        version_number,
                        version_id,
                        node_size,
                        is_level_0,
                    )?;
                    Ok(())
                },
            )
            .collect::<Result<Vec<_>, BufIoError>>()?;

        let load_time = load_start.elapsed();
        println!("Loaded regions in: {:?}", load_time);

        let lmdb = MetaDb {
            env: self.lmdb_env.clone(),
            db,
        };
        let current_version = retrieve_current_version(&lmdb)?;
        let values_range = retrieve_values_range(&lmdb)?;
        let dense_index = DenseIndex::new(
            coll.name.clone(),
            root,
            dense_index_data.levels_prob,
            dense_index_data.dim,
            prop_file.clone(),
            lmdb,
            ArcShift::new(current_version),
            ArcShift::new(dense_index_data.quantization_metric),
            ArcShift::new(dense_index_data.distance_metric),
            ArcShift::new(dense_index_data.storage_type),
            vcs,
            dense_index_data.hnsw_params,
            cache,
            index_manager,
            level_0_index_manager,
            vec_raw_manager,
            values_range.unwrap_or((-1.0, 1.0)),
            dense_index_data.sample_threshold,
            values_range.is_some(),
        );

        Ok(dense_index)
    }

    /// loads and initiates the inverted index of a collection from lmdb
    ///
    /// In doing so, the root vec for all collections'  inverted indexes are loaded into
    /// memory, which also ends up warming the cache (NodeRegistry)
    fn load_inverted_index(
        &self,
        coll: &Collection,
        root_path: &Path,
        config: &Config,
    ) -> Result<InvertedIndex, WaCustomError> {
        let collection_path: Arc<Path> = root_path.join(&coll.name).into();
        let index_path = collection_path.join("sparse_inverted_index");

        let index_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}.index", **ver)),
            8192,
        ));
        let vec_raw_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone().into(),
            |root, ver: &Hash| root.join(format!("{}.vec_raw", **ver)),
            8192,
        ));

        let db = Arc::new(
            self.lmdb_env
                .create_db(Some(&coll.name), DatabaseFlags::empty())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
        );

        let inverted_index_data =
            InvertedIndexData::load(&self.lmdb_env, self.lmdb_dense_index_db, &coll.get_key())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let vcs = Arc::new(VersionControl::from_existing(
            self.lmdb_env.clone(),
            db.clone(),
        ));
        let lmdb = MetaDb {
            env: self.lmdb_env.clone(),
            db,
        };
        let current_version = retrieve_current_version(&lmdb)?;
        let inverted_index = InvertedIndex::new(
            coll.name.clone(),
            inverted_index_data.description,
            inverted_index_data.auto_create_index,
            inverted_index_data.metadata_schema,
            inverted_index_data.max_vectors,
            lmdb,
            ArcShift::new(current_version),
            vcs,
            vec_raw_manager,
            index_manager,
            inverted_index_data.quantization,
        );

        Ok(inverted_index)
    }

    pub fn insert(&self, name: &str, dense_index: Arc<DenseIndex>) -> Result<(), WaCustomError> {
        self.inner.insert(name.to_owned(), dense_index.clone());
        persist_dense_index(
            &self.lmdb_env,
            self.lmdb_dense_index_db.clone(),
            dense_index.clone(),
        )
    }

    // TODO MERGE insert AND insert_inverted_index INTO ONE GENERIC METHOD
    // OVER INDEXES
    pub fn insert_inverted_index(
        &self,
        name: &str,
        index: Arc<InvertedIndex>,
    ) -> Result<(), WaCustomError> {
        self.inner_inverted_index
            .insert(name.to_owned(), index.clone());
        InvertedIndexData::persist(
            &self.lmdb_env,
            self.lmdb_dense_index_db.clone(),
            index.clone(),
        )
    }

    /// inserts a collection into the collections map
    #[allow(dead_code)]
    pub fn insert_collection(&self, collection: Arc<Collection>) -> Result<(), WaCustomError> {
        self.inner_collections
            .insert(collection.name.to_owned(), collection);
        Ok(())
    }

    /// Returns the `DenseIndex` by collection's name
    ///
    /// If not found, None is returned
    ///
    /// Note that it tried to look up the DenseIndex in the DashMap
    /// only and doesn't check LMDB. This is because of the assumption
    /// that at startup, all DenseIndexes will be loaded from LMDB
    /// into the in-memory DashMap and when a new DenseIndex is
    /// added, it will be written to the DashMap as well.
    ///
    /// @TODO: As a future improvement, we can fallback to checking if
    /// the DenseIndex exists in LMDB and caching it. But it's not
    /// required for the current use case.
    #[allow(dead_code)]
    pub fn get(&self, name: &str) -> Option<Arc<DenseIndex>> {
        self.inner.get(name).map(|index| index.clone())
    }

    /// Returns the `InvertedIndex` by collection's name
    ///
    /// If not found, None is returned
    ///
    /// Note that it tried to look up the InvertedIndex in the DashMap
    /// only and doesn't check LMDB. This is because of the assumption
    /// that at startup, all InvertedIndexes will be loaded from LMDB
    /// into the in-memory DashMap and when a new InvertedIndex is
    /// added, it will be written to the DashMap as well.
    ///
    /// @TODO: As a future improvement, we can fallback to checking if
    /// the InvertedIndex exists in LMDB and caching it. But it's not
    /// required for the current use case.
    pub fn get_inverted_index(&self, name: &str) -> Option<Arc<InvertedIndex>> {
        self.inner_inverted_index
            .get(name)
            .map(|index| index.clone())
    }

    /// Returns the `Collection` by collection's name
    ///
    /// If not found, None is returned
    ///
    /// Note that it tried to look up the Collections in the DashMap
    /// only and doesn't check LMDB. This is because of the assumption
    /// that at startup, all collections will be loaded from LMDB
    /// into the in-memory DashMap and when a new collection is
    /// added, it will be written to the DashMap as well.
    ///
    /// @TODO: As a future improvement, we can fallback to checking if
    /// the Collection exists in LMDB and caching it. But it's not
    /// required for the current use case.
    #[allow(dead_code)]
    pub fn get_collection(&self, name: &str) -> Option<Arc<Collection>> {
        self.inner_collections.get(name).map(|index| index.clone())
    }

    #[allow(dead_code)]
    pub fn remove(&self, name: &str) -> Result<Option<(String, Arc<DenseIndex>)>, WaCustomError> {
        match self.inner.remove(name) {
            Some((key, index)) => {
                let dense_index = delete_dense_index(
                    &self.lmdb_env,
                    self.lmdb_dense_index_db.clone(),
                    index.clone(),
                )
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
                Ok(Some((key, dense_index)))
            }
            None => Ok(None),
        }
    }

    /// removes a collection from the in-memory map
    ///
    /// returns the removed collection in case of success
    ///
    /// returns error if not found
    #[allow(dead_code)]
    pub fn remove_collection(&self, name: &str) -> Result<Arc<Collection>, WaCustomError> {
        match self.inner_collections.remove(name) {
            Some((_, collection)) => Ok(collection),
            None => {
                // collection not found, return an error response
                return Err(WaCustomError::NotFound("collection".into()));
            }
        }
    }

    #[allow(dead_code)]
    pub fn iter(
        &self,
    ) -> dashmap::iter::Iter<
        String,
        Arc<DenseIndex>,
        std::hash::RandomState,
        DashMap<String, Arc<DenseIndex>>,
    > {
        self.inner.iter()
    }

    /// returns an iterator
    #[allow(dead_code)]
    pub fn iter_collections(
        &self,
    ) -> dashmap::iter::Iter<
        String,
        Arc<Collection>,
        std::hash::RandomState,
        DashMap<String, Arc<Collection>>,
    > {
        self.inner_collections.iter()
    }
}

pub struct UsersMap {
    env: Arc<Environment>,
    users_db: Database,
    // (username, user details)
    map: DashMap<String, User>,
}

impl UsersMap {
    pub fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let users_db = env.create_db(Some("users"), DatabaseFlags::empty())?;
        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(users_db)?;
        let map = DashMap::new();

        for (username, user_bytes) in cursor.iter() {
            let username = String::from_utf8(username.to_vec()).unwrap();
            let user = User::deserialize(user_bytes).unwrap();
            map.insert(username, user);
        }

        drop(cursor);
        txn.abort();

        Ok(Self { env, users_db, map })
    }

    pub fn add_user(&self, username: String, password_hash: DoubleSHA256Hash) -> lmdb::Result<()> {
        let user = User {
            username: username.clone(),
            password_hash,
        };
        let user_bytes = user.serialize();
        let username_bytes = username.as_bytes();

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(
            self.users_db,
            &username_bytes,
            &user_bytes,
            WriteFlags::empty(),
        )?;
        txn.commit()?;

        self.map.insert(username, user);

        Ok(())
    }

    pub fn get_user(&self, username: &str) -> Option<User> {
        self.map.get(username).map(|user| user.value().clone())
    }
}

#[derive(Clone)]
pub struct User {
    pub username: String,
    pub password_hash: DoubleSHA256Hash,
}

impl User {
    fn serialize(&self) -> Vec<u8> {
        let username_bytes = self.username.as_bytes();
        let mut buf = Vec::with_capacity(32 + username_bytes.len());
        buf.extend_from_slice(&self.password_hash.0);
        buf.extend_from_slice(username_bytes);
        buf
    }

    fn deserialize(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 32 {
            return Err("Input must be at least 32 bytes".to_string());
        }
        let mut password_hash = [0u8; 32];
        password_hash.copy_from_slice(&buf[..32]);
        let username_bytes = buf[32..].to_vec();
        let username = String::from_utf8(username_bytes).map_err(|err| err.to_string())?;
        Ok(Self {
            username,
            password_hash: DoubleSHA256Hash(password_hash),
        })
    }
}

pub struct SessionDetails {
    pub created_at: u64,
    pub expires_at: u64,
    pub user: User,
}

// Define the AppEnv struct
pub struct AppEnv {
    pub collections_map: CollectionsMap,
    pub users_map: UsersMap,
    pub persist: Arc<Environment>,
    // Single hash, must not be persisted to disk, only the double hash must be
    // written to disk
    pub server_key: SingleSHA256Hash,
    pub active_sessions: Arc<DashMap<String, SessionDetails>>,
}

fn get_server_key(env: Arc<Environment>) -> lmdb::Result<SingleSHA256Hash> {
    let db = env.create_db(Some("security_metadata"), DatabaseFlags::empty())?;
    let mut txn = env.begin_rw_txn()?;
    let server_key_from_lmdb = match txn.get(db, &"server_key") {
        Ok(key) => Some(DoubleSHA256Hash(key.try_into().unwrap())),
        Err(lmdb::Error::NotFound) => None,
        Err(err) => return Err(err),
    };
    let server_key_hash = if let Some(server_key_from_lmdb) = server_key_from_lmdb {
        txn.abort();
        let entered_server_key =
            prompt_password("Enter server key: ").expect("Unable to read master key");
        let entered_server_key_hash = SingleSHA256Hash::from_str(&entered_server_key);
        let entered_server_key_double_hash = entered_server_key_hash.hash_again();
        if !server_key_from_lmdb.verify_eq(&entered_server_key_double_hash) {
            eprintln!("Invalid server key!");
            std::process::exit(1);
        }
        entered_server_key_hash
    } else {
        let entered_server_key =
            prompt_password("Create a server key: ").expect("Unable to read server key");
        let entered_server_key_hash = SingleSHA256Hash::from_str(&entered_server_key);
        let entered_server_key_double_hash = entered_server_key_hash.hash_again();
        txn.put(
            db,
            &"server_key",
            &entered_server_key_double_hash.0,
            WriteFlags::empty(),
        )?;
        txn.commit()?;
        entered_server_key_hash
    };
    Ok(server_key_hash)
}

pub fn get_app_env(config: &Config) -> Result<Arc<AppEnv>, WaCustomError> {
    let path = Path::new("./_mdb"); // TODO: prefix the customer & database name

    // Ensure the directory exists
    create_dir_all(&path).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    // Initialize the environment
    let env = Environment::new()
        .set_max_dbs(10)
        .set_map_size(1048576000) // Set the maximum size of the database to 1GB
        .open(&path)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let env_arc = Arc::new(env);

    let server_key = get_server_key(env_arc.clone())
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    let collections_map = CollectionsMap::load(env_arc.clone(), config)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let users_map = UsersMap::new(env_arc.clone())
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    // Fake user, for testing APIs
    let username = "admin".to_string();
    let password = "admin";
    let password_hash = DoubleSHA256Hash::from_str(password);
    users_map
        .add_user(username, password_hash)
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    Ok(Arc::new(AppEnv {
        collections_map,
        users_map,
        persist: env_arc,
        server_key,
        active_sessions: Arc::new(DashMap::new()),
    }))
}

#[derive(Clone)]
pub struct STM<T: 'static> {
    pub arcshift: ArcShift<T>,
    max_retries: usize,
    strict: bool,
}

fn backoff(iteration: usize) {
    let spins = 1u64 << iteration;
    for _ in 0..spins {
        std::thread::yield_now();
    }
}

impl<T> STM<T>
where
    T: 'static,
{
    pub fn new(initial_value: T, max_retries: usize, strict: bool) -> Self {
        Self {
            arcshift: ArcShift::new(initial_value),
            max_retries,
            strict,
        }
    }

    pub fn get(&mut self) -> &T {
        self.arcshift.get()
    }

    pub fn update(&mut self, new_value: T) {
        self.arcshift.update(new_value);
    }

    /// Update the value inside the ArcShift using a transactional update function.
    ///
    /// Internally it uses [ArcShift::rcu] and performs a fixed amount of retries
    /// before giving up and returning an error.
    ///
    /// TODO: Consider making the api more ergonomic. Strict and non-strict
    /// failure can be made into separate error types so that the caller
    /// does not need to check the boolean value to figure out if the
    /// update succeeded or not.
    pub fn transactional_update<F>(&mut self, mut update_fn: F) -> Result<bool, WaCustomError>
    where
        F: FnMut(&T) -> T,
    {
        let mut updated = false;
        let mut tries = 0;

        while !updated {
            // TODO: consider using rcu_maybe to avoid unnecessary updates
            // that will require changing update check semantics
            updated = self.arcshift.rcu(|t| update_fn(t));

            if !updated {
                if tries >= self.max_retries {
                    if !self.strict {
                        return Ok(false);
                    }

                    return Err(WaCustomError::LockError(
                        "Unable to update data inside ArcShift".to_string(),
                    ));
                }

                // Apply backoff before the next retry attempt
                backoff(tries);
                tries += 1;
            }
        }

        Ok(updated)
    }
}

#[derive(Debug, Clone)]
pub struct SparseVector {
    pub vector_id: u32,
    pub entries: Vec<(u32, f32)>,
}

impl SparseVector {
    pub fn new(vector_id: u32, entries: Vec<(u32, f32)>) -> Self {
        Self { vector_id, entries }
    }
}

// Dimension type for a sparse vector, based on the posting list length.
// A dimension being common or rare is relative, and is defined based on the
// query vector.
#[derive(Debug, Clone, Copy)]
pub enum SparseQueryVectorDimensionType {
    Common,
    Rare,
}

// A sparse query vector, which attaches a dimension type to each dimension
// based on the posting list length. This is used to optimize sparse ANN search
// using a cuckoo filter.
#[derive(Debug, Clone)]
pub struct SparseQueryVector {
    pub vector_id: u32,
    pub entries: Vec<(u32, SparseQueryVectorDimensionType, f32)>,
}

impl SparseQueryVector {
    pub fn new(vector_id: u32, entries: Vec<(u32, SparseQueryVectorDimensionType, f32)>) -> Self {
        Self { vector_id, entries }
    }
}
