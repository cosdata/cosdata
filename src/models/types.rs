use crate::distance::DistanceError;
use crate::distance::{
    cosine::CosineDistance, dotproduct::DotProductDistance, euclidean::EuclideanDistance,
    hamming::HammingDistance, DistanceFunction,
};
use crate::models::chunked_list::*;
use crate::models::common::*;
use crate::models::versioning::VersionHash;
use crate::quantization::product::ProductQuantization;
use crate::quantization::scalar::ScalarQuantization;
use crate::quantization::{Quantization, StorageType};
use crate::storage::Storage;
use actix_web::guard;
use bincode;
use dashmap::DashMap;
use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::fs::*;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

#[derive(Debug, Clone)]
pub struct HNSWLevel(pub u8);
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileOffset(pub u32);
pub type BytesToRead = u32;
pub type VersionId = u16;
pub type CosineSimilarity = f32;

pub type Item<T> = Arc<RwLock<T>>;

#[derive(Debug, Clone)]
pub struct Neighbour {
    pub node: LazyItem<MergedNode>,
    pub cosine_similarity: CosineSimilarity,
}

pub type PropPersistRef = (FileOffset, BytesToRead);
pub type NodeFileRef = FileOffset;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProp {
    pub id: VectorId,
    pub value: Arc<Storage>,
    pub location: Option<PropPersistRef>,
}

#[derive(Debug, Clone)]
pub enum PropState {
    Ready(Arc<NodeProp>),
    Pending(PropPersistRef),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorId {
    Str(String),
    Int(i32),
}

#[derive(Debug, Clone)]
pub struct MergedNode {
    pub version_id: VersionId,
    pub hnsw_level: HNSWLevel,
    pub prop: Item<PropState>,
    pub neighbors: LazyItems<Neighbour>,
    pub parent: Option<LazyItemRef<MergedNode>>,
    pub child: Option<LazyItemRef<MergedNode>>,
    pub versions: LazyItems<MergedNode>,
    pub persist_flag: Item<bool>,
}

#[derive(Debug)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Hamming,
    DotProduct,
}

impl DistanceFunction for DistanceMetric {
    fn calculate(&self, x: &Storage, y: &Storage) -> Result<f32, DistanceError> {
        match self {
            Self::Cosine => CosineDistance.calculate(x, y),
            Self::Euclidean => EuclideanDistance.calculate(x, y),
            Self::Hamming => HammingDistance.calculate(x, y),
            Self::DotProduct => DotProductDistance.calculate(x, y),
        }
    }
}

#[derive(Debug)]
pub enum QuantizationMetric {
    Scalar,
    Product(ProductQuantization),
}

impl Quantization for QuantizationMetric {
    fn quantize(&self, vector: &[f32], storage_type: StorageType) -> Storage {
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
    pub fn new(version_id: VersionId, hnsw_level: HNSWLevel) -> Self {
        MergedNode {
            version_id,
            hnsw_level,
            prop: Arc::new(RwLock::new(PropState::Pending((FileOffset(0), 0)))),
            neighbors: LazyItems::new(),
            parent: None,
            child: None,
            versions: LazyItems::new(),
            persist_flag: Arc::new(RwLock::new(true)),
        }
    }

    pub fn add_ready_neighbor(&self, neighbor: LazyItem<MergedNode>, cosine_similarity: f32) {
        let neighbor_ref = Arc::new(RwLock::new(Neighbour {
            node: neighbor,
            cosine_similarity,
        }));
        let lazy_item = LazyItem {
            data: Some(neighbor_ref),
            offset: None,
            decay_counter: 0,
        };
        self.neighbors.push(lazy_item);
    }

    pub fn set_parent(&mut self, parent: Option<LazyItemRef<MergedNode>>) {
        self.parent = parent;
    }

    pub fn set_child(&mut self, child: Option<LazyItemRef<MergedNode>>) {
        self.child = child;
    }

    pub fn add_ready_neighbors(&self, neighbors_list: Vec<(LazyItem<MergedNode>, f32)>) {
        for (neighbor, cosine_similarity) in neighbors_list {
            self.add_ready_neighbor(neighbor, cosine_similarity);
        }
    }

    pub fn get_neighbors(&self) -> Vec<LazyItem<Neighbour>> {
        self.neighbors.items.read().unwrap().clone()
    }

    pub fn set_neighbors(&self, new_neighbors: Vec<LazyItem<Neighbour>>) {
        let mut neighbors = self.neighbors.items.write().unwrap();
        *neighbors = new_neighbors;
    }

    pub fn add_version(&self, version: Item<MergedNode>) {
        let lazy_item = LazyItem {
            data: Some(version),
            offset: None,
            decay_counter: 0,
        };
        self.versions.push(lazy_item);
    }

    pub fn get_versions(&self) -> Vec<LazyItem<MergedNode>> {
        self.versions.items.read().unwrap().clone()
    }

    pub fn get_parent(&self) -> Option<LazyItemRef<MergedNode>> {
        self.parent.clone()
    }

    pub fn get_child(&self) -> Option<LazyItemRef<MergedNode>> {
        self.child.clone()
    }

    pub fn set_prop_location(&self, new_location: PropPersistRef) {
        let mut prop = self.prop.write().unwrap();
        *prop = PropState::Pending(new_location);
    }

    pub fn get_prop_location(&self) -> Option<PropPersistRef> {
        let prop = self.prop.read().unwrap();
        match *prop {
            PropState::Ready(ref node_prop) => node_prop.location,
            PropState::Pending(location) => Some(location),
        }
    }

    pub fn get_prop(&self) -> PropState {
        self.prop.read().unwrap().clone()
    }

    pub fn set_prop_pending(&self, prop_ref: PropPersistRef) {
        let mut prop = self.prop.write().unwrap();
        *prop = PropState::Pending(prop_ref);
    }

    pub fn set_prop_ready(&self, node_prop: Arc<NodeProp>) {
        let mut prop = self.prop.write().unwrap();
        *prop = PropState::Ready(node_prop);
    }

    pub fn set_persistence(&self, flag: bool) {
        let mut fl = self.persist_flag.write().unwrap();
        *fl = flag;
    }

    pub fn needs_persistence(&self) -> bool {
        let fl = self.persist_flag.read().unwrap();
        *fl
    }
}

impl SyncPersist for MergedNode {
    fn set_persistence(&self, flag: bool) {
        let mut fl = self.persist_flag.write().unwrap();
        *fl = flag;
    }

    fn needs_persistence(&self) -> bool {
        let fl = self.persist_flag.read().unwrap();
        *fl
    }
}

impl SyncPersist for Neighbour {
    fn set_persistence(&self, flag: bool) {
        let Some(node) = self.node.data.clone() else {
            return;
        };
        let guard = node.read().unwrap();
        let mut fl = guard.persist_flag.write().unwrap();
        *fl = flag;
    }

    fn needs_persistence(&self) -> bool {
        let Some(node) = self.node.data.clone() else {
            return false;
        };
        let guard = node.read().unwrap();
        let fl = guard.persist_flag.read().unwrap();
        *fl
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
impl fmt::Display for MergedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MergedNode {{ version_id: {}, hnsw_level: {}, prop: {:?}, neighbors: {:?}, parent: {:?}, child: {:?}, version_ref: {:?} }}",
            self.version_id,
            self.hnsw_level.0,
            self.prop.read().unwrap(),
            self.neighbors,
            self.parent,
            self.child,
            self.versions
        )
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
        let quant_vec = simp_quant(vec);
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

pub type SizeBytes = u32;

// needed to flatten and get uniques
pub type ExecQueueUpdate = Item<Vec<LazyItem<MergedNode>>>;

#[derive(Debug, Clone)]
pub struct MetaDb {
    pub env: Arc<Environment>,
    pub db: Arc<Database>,
}

#[derive(Debug, Clone)]
pub struct VectorStore {
    pub exec_queue_nodes: ExecQueueUpdate,
    pub max_cache_level: u8,
    pub database_name: String,
    pub root_vec: LazyItemRef<MergedNode>,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub quant_dim: usize,
    pub prop_file: Arc<File>,
    pub version_lmdb: MetaDb,
    pub current_version: Item<Option<VersionHash>>,
    pub current_open_transaction: Item<Option<VersionHash>>,
    pub quantization_metric: Arc<QuantizationMetric>,
    pub distance_metric: Arc<DistanceMetric>,
    pub storage_type: StorageType,
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
        version_lmdb: MetaDb,
        current_version: Item<Option<VersionHash>>,
        quantization_metric: Arc<QuantizationMetric>,
        distance_metric: Arc<DistanceMetric>,
        storage_type: StorageType,
    ) -> Self {
        VectorStore {
            exec_queue_nodes,
            max_cache_level,
            database_name,
            root_vec,
            levels_prob,
            quant_dim,
            prop_file,
            version_lmdb,
            current_version,
            current_open_transaction: Arc::new(RwLock::new(None)),
            quantization_metric,
            distance_metric,
            storage_type,
        }
    }
    // Get method
    pub fn get_current_version(
        &self,
    ) -> Result<
        Option<VersionHash>,
        std::sync::PoisonError<std::sync::RwLockReadGuard<'_, Option<VersionHash>>>,
    > {
        self.current_version.read().map(|guard| guard.clone())
    }

    // Set method
    pub fn set_current_version(
        &self,
        new_version: Option<VersionHash>,
    ) -> Result<(), std::sync::PoisonError<std::sync::RwLockWriteGuard<'_, Option<VersionHash>>>>
    {
        let mut write_guard = self.current_version.write()?;
        *write_guard = new_version;
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct VectorEmbedding {
    pub raw_vec: Arc<Storage>,
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
                .set_max_dbs(1)
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
