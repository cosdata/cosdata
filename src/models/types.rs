use dashmap::DashMap;
use rocksdb::{ColumnFamily, Error, Options, DB};
use std::sync::{Arc, Mutex};

pub type NumericVector = Vec<f32>;
pub type VectorHash = Vec<u8>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum VectorId {
    Str(String),
    Int(i32),
}

type CacheType = DashMap<(i8, VectorId), Option<Arc<(VectorTreeNode)>>>;

#[derive(Debug, Clone)]
pub struct VectorStore {
    pub cache: Arc<CacheType>,
    pub max_cache_level: i8,
    pub database_name: String,
    pub root_vec: (VectorId, NumericVector),
}

#[derive(Debug, Clone)]
pub struct VectorEmbedding {
    pub raw_vec: Arc<NumericVector>,
    pub hash_vec: VectorId,
}

#[derive(Debug, Clone)]
pub struct VectorTreeNode {
    pub vector_list: Arc<NumericVector>,
    pub neighbors: Vec<(VectorId, f32)>,
}

type VectorStoreMap = DashMap<String, VectorStore>;
type UserDataCache = DashMap<String, (String, i32, i32, std::time::SystemTime, Vec<String>)>;

// Define the AppEnv struct
pub struct AppEnv {
    pub user_data_cache: UserDataCache,
    pub vector_store_map: VectorStoreMap,
    pub persist: Arc<Mutex<Persist>>,
}
// Initialize the AppEnv with once_cell
use once_cell::sync::OnceCell;

use super::{common::WaCustomError, persist::Persist};
static AIN_ENV: OnceCell<Result<Arc<AppEnv>, WaCustomError>> = OnceCell::new();

pub fn get_app_env() -> Result<Arc<AppEnv> , WaCustomError>{
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
