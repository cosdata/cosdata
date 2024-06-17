use bincode;
use dashmap::DashMap;
use rocksdb::{ColumnFamily, Error, Options, DB};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub type HNSWLevel = u8;

pub type NodeRef = Arc<Node>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeighbourRef {
    Ready {
        node: NodeRef,
        cosine_similarity: f32,
    },
    Pending(NodeFileRef),
}

pub type NodeFileRef = (u32, u32); // (file_number, offset)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProp {
    pub id: VectorId,
    pub value: Arc<VectorQt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub prop: Arc<NodeProp>,
    pub location: Option<NodeFileRef>,
    pub neighbors: Arc<RwLock<Vec<NeighbourRef>>>,
    pub parent: Arc<RwLock<Option<NodeRef>>>,
    pub child: Arc<RwLock<Option<NodeRef>>>,
}

impl NodeProp {
    pub fn new(id: VectorId, value: Arc<VectorQt>) -> Arc<NodeProp> {
        Arc::new(NodeProp { id, value })
    }
}

impl Node {
    pub fn new(prop: Arc<NodeProp>, loc: Option<NodeFileRef>) -> NodeRef {
        Arc::new(Node {
            prop,
            neighbors: Arc::new(RwLock::new(Vec::new())),
            location: loc,
            parent: Arc::new(RwLock::new(None)),
            child: Arc::new(RwLock::new(None)),
        })
    }

    pub fn add_ready_neighbor(&self, neighbor: NodeRef, cosine_similarity: f32) {
        let mut neighbors = self.neighbors.write().unwrap();
        neighbors.push(NeighbourRef::Ready {
            node: neighbor,
            cosine_similarity,
        });
    }

    pub fn add_ready_neighbors(&self, neighbors_list: Vec<(NodeRef, f32)>) {
        let mut neighbors = self.neighbors.write().unwrap();
        for (neighbor, cosine_similarity) in neighbors_list.iter() {
            neighbors.push(NeighbourRef::Ready {
                node: neighbor.clone(),
                cosine_similarity: *cosine_similarity,
            });
        }
    }

    pub fn get_neighbors(&self) -> Vec<NeighbourRef> {
        let neighbors = self.neighbors.read().unwrap();
        neighbors.clone()
    }

    pub fn add_parent(&self, parent: NodeRef) {
        let mut parent_lock = self.parent.write().unwrap();
        *parent_lock = Some(parent);
    }

    pub fn add_child(&self, child: NodeRef) {
        let mut child_lock = self.child.write().unwrap();
        *child_lock = Some(child);
    }

    pub fn get_parent(&self) -> Option<NodeRef> {
        let parent_lock = self.parent.read().unwrap();
        parent_lock.clone()
    }

    pub fn get_child(&self) -> Option<NodeRef> {
        let child_lock = self.child.read().unwrap();
        child_lock.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQt {
    pub mag: f64,
    pub quant_vec: Vec<u32>,
    pub resolution: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorId {
    Str(String),
    Int(i32),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorTreeNode {
    pub vector_list: Arc<VectorQt>,
    pub neighbors: Vec<(VectorId, f32)>,
}

impl VectorTreeNode {
    // Serialize the VectorTreeNode to a byte vector
    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(self)?;
        Ok(serialized)
    }

    // Deserialize a byte vector to a VectorTreeNode
    pub fn deserialize(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let deserialized = bincode::deserialize(bytes)?;
        Ok(deserialized)
    }
}
type CacheType = DashMap<(HNSWLevel, VectorId), NodeRef>;

#[derive(Debug, Clone)]
pub struct VectorStore {
    pub cache: Arc<CacheType>,
    pub max_cache_level: u8,
    pub database_name: String,
    pub root_vec: (VectorId, VectorQt),
    pub levels_prob: Arc<Vec<(f64, i32)>>,
    pub quant_dim: usize,
}

#[derive(Debug, Clone)]
pub struct VectorEmbedding {
    pub raw_vec: Arc<VectorQt>,
    pub hash_vec: VectorId,
}

type VectorStoreMap = DashMap<String, VectorStore>;
type UserDataCache = DashMap<String, (String, i32, i32, std::time::SystemTime, Vec<String>)>;

// Define the AppEnv struct
pub struct AppEnv {
    pub user_data_cache: UserDataCache,
    pub vector_store_map: VectorStoreMap,
    pub persist: Arc<Mutex<Persist>>,
}

use super::{common::WaCustomError, persist::Persist};
static AIN_ENV: OnceLock<Result<Arc<AppEnv>, WaCustomError>> = OnceLock::new();

pub fn get_app_env() -> Result<Arc<AppEnv>, WaCustomError> {
    AIN_ENV
        .get_or_init(|| {
            let path = "./xdb/"; // Change this to your desired path
            let result = match Persist::new(path) {
                Ok(value) => Ok(Arc::new(AppEnv {
                    user_data_cache: DashMap::new(),
                    vector_store_map: DashMap::new(),
                    persist: Arc::new(Mutex::new(value)),
                })),
                Err(e) => Err(WaCustomError::CreateDatabaseFailed(e.to_string())),
            };
            result
        })
        .clone()
}

// impl PartialEq for VectorId {
//     fn eq(&self, other: &Self) -> bool {
//         match (self, other) {
//             (VectorId::Str(s1), VectorId::Str(s2)) => s1 == s2,
//             (VectorId::Int(i1), VectorId::Int(i2)) => i1 == i2,
//             _ => false,
//         }
//     }
// }

// impl Eq for VectorId {}

// impl Hash for VectorId {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         match self {
//             VectorId::Str(s) => s.hash(state),
//             VectorId::Int(i) => i.hash(state),
//         }
//     }
// }
