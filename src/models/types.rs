use crate::models::chunked_list::*;
use crate::models::common::*;
use crate::models::versioning::VersionHash;
use bincode;
use dashmap::DashMap;
use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::*;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub type HNSWLevel = u8;
pub type FileOffset = u32;
pub type BytesToRead = u32;
pub type VersionId = u16;
pub type CosineSimilarity = f32;

#[derive(Debug, Clone)]
pub struct Neighbour {
    pub node: Arc<MergedNode>,
    pub cosine_similarity: CosineSimilarity,
}

pub type PropPersistRef = (FileOffset, BytesToRead);
pub type NodeFileRef = FileOffset;

#[derive(Debug, Clone)]
pub struct NodeProp {
    pub id: VectorId,
    pub value: Arc<VectorQt>,
    pub location: Option<PropPersistRef>,
}

#[derive(Debug, Clone)]
pub enum PropState {
    Ready(Arc<NodeProp>),
    Pending(PropPersistRef),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorId {
    Str(String),
    Int(i32),
}

#[derive(Debug, Clone)]
pub struct MergedNode {
    pub version_id: VersionId,
    pub hnsw_level: HNSWLevel,
    pub prop: Arc<RwLock<PropState>>,
    pub location: Arc<RwLock<Option<FileOffset>>>,
    pub neighbors: Arc<RwLock<ItemListRef<Neighbour>>>,
    pub parent: Arc<RwLock<LazyItem<MergedNode>>>,
    pub child: Arc<RwLock<LazyItem<MergedNode>>>,
    pub version_ref: Arc<RwLock<ItemListRef<MergedNode>>>,
    persist_flag: Arc<RwLock<bool>>,
}

impl Locatable for MergedNode {
    fn get_file_offset(&self) -> Option<u32> {
        self.get_location()
    }

    fn set_file_offset(&mut self, location: u32) {
        self.set_location(location);
    }

    fn set_persistence(&mut self, flag: bool) {
        let mut fl = self.persist_flag.write().unwrap();
        *fl = flag;
    }

    fn needs_persistence(&self) -> bool {
        let fl = self.persist_flag.read().unwrap();
        *fl
    }
}

impl Locatable for Neighbour {
    fn get_file_offset(&self) -> Option<u32> {
        self.node.get_location()
    }

    fn set_file_offset(&mut self, location: u32) {
        self.node.set_location(location);
    }

    fn set_persistence(&mut self, flag: bool) {
        let mut fl = self.node.persist_flag.write().unwrap();
        *fl = flag;
    }

    fn needs_persistence(&self) -> bool {
        let fl = self.node.persist_flag.read().unwrap();
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

impl MergedNode {
    pub fn new(version_id: VersionId, hnsw_level: HNSWLevel) -> Self {
        MergedNode {
            version_id,
            hnsw_level,
            prop: Arc::new(RwLock::new(PropState::Pending((0, 0)))),
            location: Arc::new(RwLock::new(None)),
            neighbors: Arc::new(RwLock::new(ItemListRef::Null)),
            parent: Arc::new(RwLock::new(None)),
            child: Arc::new(RwLock::new(None)),
            version_ref: Arc::new(RwLock::new(ItemListRef::Null)),
            persist_flag: Arc::new(RwLock::new(true)),
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

    pub fn add_ready_neighbor(&self, neighbor: Arc<MergedNode>, cosine_similarity: f32) {
        let mut neighbors = self.neighbors.write().unwrap();
        let neighbor_ref = Arc::new(Neighbour {
            node: neighbor,
            cosine_similarity,
        });
        neighbors.add(LazyItem::Ready(neighbor_ref));
    }

    pub fn add_ready_neighbors(&self, neighbors_list: Vec<(Arc<MergedNode>, f32)>) {
        let mut neighbors = self.neighbors.write().unwrap();
        for (neighbor, cosine_similarity) in neighbors_list {
            let neighbor_ref = Arc::new(Neighbour {
                node: neighbor,
                cosine_similarity,
            });
            neighbors.add(LazyItem::Ready(neighbor_ref));
        }
    }

    pub fn get_neighbors(&self) -> Vec<LazyItem<Neighbour>> {
        let neighbors = self.neighbors.read().unwrap();
        neighbors.get_all()
    }

    pub fn set_neighbors(&self, new_neighbors: Vec<Neighbour>) {
        let mut neighbors = self.neighbors.write().unwrap();
        let new_neighbors_refs = new_neighbors
            .into_iter()
            .map(|nr| LazyItem::Ready(Arc::new(nr)))
            .collect();
        neighbors.set_all(new_neighbors_refs);
    }

    pub fn add_version(&self, version: Arc<MergedNode>) {
        let mut versions = self.version_ref.write().unwrap();
        versions.add(LazyItem::Ready(version));
    }

    pub fn get_versions(&self) -> Vec<LazyItem<MergedNode>> {
        let versions = self.version_ref.read().unwrap();
        versions.get_all()
    }

    pub fn set_parent(&self, parent: Arc<MergedNode>) {
        let mut parent_lock = self.parent.write().unwrap();
        *parent_lock = Some(parent);
    }

    pub fn set_child(&self, child: Arc<MergedNode>) {
        let mut child_lock = self.child.write().unwrap();
        *child_lock = Some(child);
    }

    pub fn get_parent(&self) -> Option<Arc<MergedNode>> {
        let parent_lock = self.parent.read().unwrap();
        parent_lock.clone()
    }

    pub fn get_child(&self) -> Option<Arc<MergedNode>> {
        let child_lock = self.child.read().unwrap();
        child_lock.clone()
    }

    pub fn set_location(&self, new_location: NodeFileRef) {
        let mut location_write = self.location.write().unwrap();
        *location_write = Some(new_location);
    }

    pub fn get_location(&self) -> Option<NodeFileRef> {
        let location_read = self.location.read().unwrap();
        *location_read
    }

    pub fn set_prop_location(&self, new_location: PropPersistRef) {
        let mut location_write = self.prop.write().unwrap();
        *location_write = PropState::Pending(new_location);
    }

    pub fn get_prop_location(&self) -> Option<PropPersistRef> {
        let location_read = self.prop.read().unwrap();
        match location_read.clone() {
            PropState::Ready(x) => x.clone().location,
            PropState::Pending(x) => Some(x),
        }
    }
}
impl fmt::Display for MergedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MergedNode {{ version_id: {}, hnsw_level: {}, prop: {:?}, location: {:?}, neighbors: {:?}, parent: {:?}, child: {:?}, version_ref: {:?} }}",
            self.version_id,
            self.hnsw_level,
            self.prop.read().unwrap(),
            self.location.read().unwrap(),
            self.neighbors.read().unwrap(),
            self.parent.read().unwrap(),
            self.child.read().unwrap(),
            self.version_ref.read().unwrap()
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
        quant_vec: Vec<Vec<u32>>,
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
        let quant_vec = quantize_to_u32_bits(vec, resolution);
        let mag = 0; //implement a proper magnitude calculation
        Self::SubByte {
            mag,
            quant_vec,
            resolution,
        }
    }
}

// #[derive(Debug, Clone)]
// pub struct VectorTreeNode {
//     pub vector_list: Arc<VectorQt>,
//     pub neighbors: Vec<(VectorId, f32)>,
// }

// impl VectorTreeNode {
//     // Serialize the VectorTreeNode to a byte vector
//     pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//         let serialized = bincode::serialize(self)?;
//         Ok(serialized)
//     }

//     // Deserialize a byte vector to a VectorTreeNode
//     pub fn deserialize(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
//         let deserialized = bincode::deserialize(bytes)?;
//         Ok(deserialized)
//     }
// }

pub type SizeBytes = u32;

// needed to flatten and get uniques
pub type ExecQueueUpdate = Arc<DashMap<(HNSWLevel, VectorId), (Arc<MergedNode>, SizeBytes)>>;

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
    pub root_vec: Arc<MergedNode>,
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub quant_dim: usize,
    pub prop_file: Arc<File>,
    pub wal_file: Arc<File>,
    pub version_lmdb: MetaDb,
    pub current_version: Arc<RwLock<Option<VersionHash>>>,
}
impl VectorStore {
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
    pub raw_vec: Arc<VectorQt>,
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
            create_dir_all(&path)
                .map_err(|e| WaCustomError::CreateDatabaseFailed(e.to_string()))?;
            // Initialize the environment
            let env = Environment::new()
                .set_max_dbs(1)
                .set_map_size(10485760) // Set the maximum size of the database to 10MB
                .open(&path)
                .map_err(|e| WaCustomError::CreateDatabaseFailed(e.to_string()))?;

            Ok(Arc::new(AppEnv {
                user_data_cache: DashMap::new(),
                vector_store_map: DashMap::new(),
                persist: Arc::new(env),
            }))
        })
        .clone()
}
