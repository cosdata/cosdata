use super::buffered_io::BufferManagerFactory;
use super::cache_loader::NodeRegistry;
use super::collection::Collection;
use super::meta_persist::{
    delete_dense_index, lmdb_init_collections_db, lmdb_init_db, load_collections,
    load_dense_index_data, persist_dense_index, retrieve_current_version,
};
use super::serializer::CustomSerialize;
use super::versioning::VersionControl;
use crate::distance::cosine::CosineSimilarity;
use crate::distance::DistanceError;
use crate::distance::{
    cosine::CosineDistance, dotproduct::DotProductDistance, euclidean::EuclideanDistance,
    hamming::HammingDistance, DistanceFunction,
};
use crate::models::common::*;
use crate::models::identity_collections::*;
use crate::models::lazy_load::*;
use crate::models::versioning::*;
use crate::quantization::{
    product::ProductQuantization, scalar::ScalarQuantization, Quantization, QuantizationError,
    StorageType,
};
use crate::storage::Storage;
use arcshift::ArcShift;
use dashmap::DashMap;
use lmdb::{Database, DatabaseFlags, Environment};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::fs::*;
use std::hash::{DefaultHasher, Hash as StdHash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HNSWLevel(pub u8);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileOffset(pub u32);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, Hash)]
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

#[derive(Debug, Clone)]
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
pub enum VectorId {
    Str(String),
    Int(i32),
}

#[derive(Clone)]
pub struct MergedNode {
    pub hnsw_level: HNSWLevel,
    pub prop: ArcShift<PropState>,
    pub neighbors: EagerLazyItemSet<MergedNode, MetricResult>,
    pub parent: LazyItemRef<MergedNode>,
    pub child: LazyItemRef<MergedNode>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum MetricResult {
    CosineSimilarity(CosineSimilarity),
    CosineDistance(CosineDistance),
    EuclideanDistance(EuclideanDistance),
    HammingDistance(HammingDistance),
    DotProductDistance(DotProductDistance),
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
    ) -> Result<Storage, QuantizationError> {
        match self {
            Self::Scalar => ScalarQuantization.quantize(vector, storage_type),
            Self::Product(product) => product.quantize(vector, storage_type),
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
        match self {
            VectorId::Str(s) => write!(f, "{}", s),
            VectorId::Int(i) => write!(f, "{}", i),
        }
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

pub struct SizeBytes(pub u32);

// needed to flatten and get uniques
pub type ExecQueueUpdate = STM<Vec<ArcShift<LazyItem<MergedNode>>>>;

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

#[derive(Debug, Clone)]
pub struct HNSWHyperParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub num_layers: u8,
    pub max_cache_size: usize,
}

impl Default for HNSWHyperParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            ef_search: 15,
            num_layers: 5,
            max_cache_size: 1000,
        }
    }
}

#[derive(Clone)]
pub struct DenseIndex {
    pub exec_queue_nodes: ExecQueueUpdate,
    pub database_name: String,
    pub root_vec: LazyItemRef<MergedNode>,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub dim: usize,
    pub prop_file: Arc<File>,
    pub lmdb: MetaDb,
    pub current_version: ArcShift<Hash>,
    pub current_open_transaction: ArcShift<Option<Hash>>,
    pub quantization_metric: ArcShift<QuantizationMetric>,
    pub distance_metric: ArcShift<DistanceMetric>,
    pub storage_type: ArcShift<StorageType>,
    pub vcs: Arc<VersionControl>,
    pub hnsw_params: ArcShift<HNSWHyperParams>,
    // Whether the VectorStore has been configured or not
    pub configured: Arc<AtomicBool>,
    pub auto_config: Arc<AtomicBool>,
    pub cache: Arc<NodeRegistry>,
    pub index_manager: Arc<BufferManagerFactory>,
    pub vec_raw_manager: Arc<BufferManagerFactory>,
}

impl DenseIndex {
    pub fn new(
        exec_queue_nodes: ExecQueueUpdate,
        database_name: String,
        root_vec: LazyItemRef<MergedNode>,
        levels_prob: Arc<Vec<(f64, i32)>>,
        dim: usize,
        prop_file: Arc<File>,
        lmdb: MetaDb,
        current_version: ArcShift<Hash>,
        quantization_metric: ArcShift<QuantizationMetric>,
        distance_metric: ArcShift<DistanceMetric>,
        storage_type: ArcShift<StorageType>,
        vcs: Arc<VersionControl>,
        num_layers: u8,
        auto_config: bool,
        cache: Arc<NodeRegistry>,
        index_manager: Arc<BufferManagerFactory>,
        vec_raw_manager: Arc<BufferManagerFactory>,
    ) -> Self {
        DenseIndex {
            exec_queue_nodes,
            database_name,
            root_vec,
            levels_prob,
            dim,
            prop_file,
            lmdb,
            current_version,
            current_open_transaction: ArcShift::new(None),
            quantization_metric,
            distance_metric,
            storage_type,
            vcs,
            hnsw_params: ArcShift::new(HNSWHyperParams {
                num_layers,
                ..Default::default()
            }),
            configured: Arc::new(AtomicBool::new(false)),
            auto_config: Arc::new(AtomicBool::new(auto_config)),
            cache,
            index_manager,
            vec_raw_manager,
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

    pub fn get_configured_flag(&self) -> bool {
        self.configured.load(Ordering::Relaxed)
    }

    pub fn set_configured_flag(&self, flag: bool) {
        self.configured.store(flag, Ordering::Relaxed);
    }

    pub fn get_auto_config_flag(&self) -> bool {
        self.auto_config.load(Ordering::Relaxed)
    }

    pub fn set_auto_config_flag(&self, flag: bool) {
        self.auto_config.store(flag, Ordering::Relaxed);
    }

    /// Returns FileIndex (offset) corresponding to the root
    /// node. Returns None if the it's not set or the root node is an
    /// invalid LazyItem
    pub fn root_vec_offset(&self) -> Option<FileIndex> {
        let lazy_item = self.root_vec.item.shared_get();
        match lazy_item {
            LazyItem::Valid { file_index, .. } => file_index.shared_get().clone(),
            LazyItem::Invalid => None,
        }
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
    pub raw_vec: Vec<f32>,
    pub hash_vec: VectorId,
}

pub(crate) struct CollectionsMap {
    /// holds an in-memory map of all dense indexes for all collections
    inner: DashMap<String, Arc<DenseIndex>>,
    inner_collections: DashMap<String, Arc<Collection>>,
    lmdb_env: Arc<Environment>,
    // made it public temporarily
    // just to be able to persist collections from outside CollectionsMap
    pub(crate) lmdb_collections_db: Database,
    lmdb_dense_index_db: Database,
    lmdb_inverted_index_db: Database,
}

impl CollectionsMap {
    fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let collections_db = lmdb_init_collections_db(&env)?;
        let dense_index_db = lmdb_init_db(&env, "dense_indexes")?;
        let inverted_index_db = lmdb_init_db(&env, "inverted_indexes")?;
        let res = Self {
            inner: DashMap::new(),
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
    fn load(env: Arc<Environment>) -> Result<Self, WaCustomError> {
        let collections_map =
            Self::new(env.clone()).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collections = load_collections(
            &collections_map.lmdb_env,
            collections_map.lmdb_collections_db.clone(),
        )
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let root_path = Path::new(".");

        // let bufmans = cache.get_bufmans();

        for coll in collections {
            let coll = Arc::new(coll);
            collections_map
                .inner_collections
                .insert(coll.name.clone(), coll.clone());

            // if collection has dense index load it from the lmdb
            if coll.dense_vector.enabled {
                let dense_index = collections_map.load_dense_index(&coll, root_path)?;
                collections_map
                    .inner
                    .insert(coll.name.clone(), Arc::new(dense_index));
            }

            // if collection has inverted index load it from the lmdb
            if coll.sparse_vector.enabled {
                println!("inverted index should be loaded here!")
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
    ) -> Result<DenseIndex, WaCustomError> {
        let collection_path: Arc<Path> = root_path.join(&coll.name).into();

        let index_manager = Arc::new(BufferManagerFactory::new(
            collection_path.clone(),
            |root, ver| root.join(format!("{}.index", **ver)),
        ));
        let vec_raw_manager = Arc::new(BufferManagerFactory::new(
            collection_path.clone(),
            |root, ver| root.join(format!("{}.vec_raw", **ver)),
        ));
        // TODO: May be the value can be taken from config
        let cache = Arc::new(NodeRegistry::new(1000, index_manager.clone()));

        let db = Arc::new(
            self.lmdb_env
                .create_db(Some(&coll.name), DatabaseFlags::empty())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?,
        );

        let dense_index_data =
            load_dense_index_data(&self.lmdb_env, self.lmdb_dense_index_db, &coll.get_key())
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let root_item: LazyItem<MergedNode> = LazyItem::deserialize(
            index_manager.clone(),
            dense_index_data.file_index,
            cache.clone(),
            1000,
            &mut HashSet::new(),
        )?;
        let root = LazyItemRef::from_lazy(root_item);

        let vcs = Arc::new(VersionControl::from_existing(
            self.lmdb_env.clone(),
            db.clone(),
        ));
        let prop_file = Arc::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(collection_path.join("prop.data"))
                .unwrap(),
        );
        let lmdb = MetaDb {
            env: self.lmdb_env.clone(),
            db,
        };
        let current_version = retrieve_current_version(&lmdb)?;
        let dense_index = DenseIndex::new(
            STM::new(Vec::new(), 16, true),
            // dense_index_data.max_level,
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
            dense_index_data.num_layers,
            // TODO: persist
            true,
            cache,
            index_manager,
            vec_raw_manager,
        );

        Ok(dense_index)
    }

    pub fn insert(&self, name: &str, dense_index: Arc<DenseIndex>) -> Result<(), WaCustomError> {
        self.inner.insert(name.to_owned(), dense_index.clone());
        persist_dense_index(
            &self.lmdb_env,
            self.lmdb_dense_index_db.clone(),
            dense_index.clone(),
        )
    }

    /// inserts a collection into the collections map
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
    pub fn get(&self, name: &str) -> Option<Arc<DenseIndex>> {
        self.inner.get(name).map(|index| index.clone())
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
    pub fn get_collection(&self, name: &str) -> Option<Arc<Collection>> {
        self.inner_collections.get(name).map(|index| index.clone())
    }

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
    pub fn remove_collection(&self, name: &str) -> Result<Arc<Collection>, WaCustomError> {
        match self.inner_collections.remove(name) {
            Some((_, collection)) => Ok(collection),
            None => {
                // collection not found, return an error response
                return Err(WaCustomError::NotFound("collection".into()));
            }
        }
    }

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

type UserDataCache = DashMap<String, (String, i32, i32, std::time::SystemTime, Vec<String>)>;

// Define the AppEnv struct
pub struct AppEnv {
    pub user_data_cache: UserDataCache,
    pub collections_map: CollectionsMap,
    pub persist: Arc<Environment>,
}

pub fn get_app_env() -> Result<Arc<AppEnv>, WaCustomError> {
    let path = Path::new("./_mdb"); // TODO: prefix the customer & database name

    // Ensure the directory exists
    create_dir_all(&path).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    // Initialize the environment
    let env = Environment::new()
        .set_max_dbs(10)
        .set_map_size(10485760) // Set the maximum size of the database to 10MB
        .open(&path)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let env_arc = Arc::new(env);

    let collections_map = CollectionsMap::load(env_arc.clone())
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    Ok(Arc::new(AppEnv {
        user_data_cache: DashMap::new(),
        collections_map,
        persist: env_arc,
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
