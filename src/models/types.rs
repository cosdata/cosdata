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
use crate::quantization::product::ProductQuantization;
use crate::quantization::scalar::ScalarQuantization;
use crate::quantization::{Quantization, QuantizationError, StorageType};
use crate::storage::Storage;
use arcshift::ArcShift;
use dashmap::DashMap;
use lmdb::{Database, Environment};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::*;
use std::hash::{DefaultHasher, Hash as StdHash, Hasher};
use std::path::Path;
use std::sync::{Arc, OnceLock};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProp {
    pub id: VectorId,
    pub value: Arc<Storage>,
    pub location: Option<PropPersistRef>,
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

#[derive(Debug, serde::Deserialize, serde::Serialize)]
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

#[derive(Debug)]
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

    fn train(
        &mut self,
        vectors: &[Vec<f32>],
    ) -> Result<(), crate::quantization::QuantizationError> {
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

    pub fn set_prop_location(&self, new_location: PropPersistRef) {
        let mut arc = self.prop.clone();
        arc.rcu(|prop| match prop {
            PropState::Pending(_) => PropState::Pending(new_location),
            PropState::Ready(prop) => {
                let mut new_prop = NodeProp::clone(&prop);
                new_prop.location = Some(new_location);
                PropState::Ready(Arc::new(new_prop))
            }
        });
    }

    pub fn get_prop_location(&self) -> Option<PropPersistRef> {
        let mut arc = self.prop.clone();
        match arc.get() {
            PropState::Ready(ref node_prop) => node_prop.location,
            PropState::Pending(location) => Some(*location),
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
    pub metadata_db: Arc<Database>,
    pub embeddings_db: Arc<Database>,
}

#[derive(Clone)]
pub struct VectorStore {
    pub exec_queue_nodes: ExecQueueUpdate,
    pub max_cache_level: u8,
    pub database_name: String,
    pub root_vec: LazyItemRef<MergedNode>,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub quant_dim: usize,
    pub prop_file: Arc<File>,
    pub lmdb: MetaDb,
    pub current_version: ArcShift<Hash>,
    pub current_open_transaction: ArcShift<Option<Hash>>,
    pub quantization_metric: Arc<QuantizationMetric>,
    pub distance_metric: Arc<DistanceMetric>,
    pub storage_type: StorageType,
    pub vcs: Arc<VersionControl>,
}

impl VectorStore {
    pub fn new(
        exec_queue_nodes: ExecQueueUpdate,
        max_cache_level: u8,
        database_name: String,
        root_vec: LazyItemRef<MergedNode>,
        levels_prob: Arc<Vec<(f64, i32)>>,
        quant_dim: usize,
        prop_file: Arc<File>,
        lmdb: MetaDb,
        current_version: ArcShift<Hash>,
        quantization_metric: Arc<QuantizationMetric>,
        distance_metric: Arc<DistanceMetric>,
        storage_type: StorageType,
        vcs: Arc<VersionControl>,
    ) -> Self {
        VectorStore {
            exec_queue_nodes,
            max_cache_level,
            database_name,
            root_vec,
            levels_prob,
            quant_dim,
            prop_file,
            lmdb,
            current_version,
            current_open_transaction: ArcShift::new(None),
            quantization_metric,
            distance_metric,
            storage_type,
            vcs,
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

type VectorStoreMap = DashMap<String, Arc<VectorStore>>;
type UserDataCache = DashMap<String, (String, i32, i32, std::time::SystemTime, Vec<String>)>;

// Define the AppEnv struct
pub struct AppEnv {
    pub user_data_cache: UserDataCache,
    pub vector_store_map: VectorStoreMap,
    pub persist: Arc<Environment>,
}

static AIN_ENV: OnceLock<Result<Arc<AppEnv>, WaCustomError>> = OnceLock::new();

pub fn get_app_env() -> Result<Arc<AppEnv>, WaCustomError> {
    AIN_ENV
        .get_or_init(|| {
            let path = Path::new("./_mdb"); // TODO: prefix the customer & database name

            // Ensure the directory exists
            create_dir_all(&path).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
            // Initialize the environment
            let env = Environment::new()
                .set_max_dbs(4)
                .set_map_size(10485760) // Set the maximum size of the database to 10MB
                .open(&path)
                .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

            Ok(Arc::new(AppEnv {
                user_data_cache: DashMap::new(),
                vector_store_map: DashMap::new(),
                persist: Arc::new(env),
            }))
        })
        .clone()
}

#[derive(Clone)]
pub struct STM<T: 'static> {
    pub arcshift: ArcShift<T>,
    max_retries: usize,
    strict: bool,
}

fn backoff(iteration: usize) {
    let spins = 1 << iteration;
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
