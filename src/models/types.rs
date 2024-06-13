use bincode;
use dashmap::DashMap;
use rocksdb::{ColumnFamily, Error, Options, DB};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

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
type CacheType = DashMap<(i8, VectorId), Option<Arc<(VectorTreeNode)>>>;

#[derive(Debug, Clone)]
pub struct VectorStore {
    pub cache: Arc<CacheType>,
    pub max_cache_level: i8,
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
