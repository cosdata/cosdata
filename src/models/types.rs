use dashmap::DashMap;
use std::sync::Arc;

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
}
// Initialize the AppEnv with once_cell
use once_cell::sync::OnceCell;
static AIN_ENV: OnceCell<Arc<AppEnv>> = OnceCell::new();

pub fn get_app_env() -> Arc<AppEnv> {
    AIN_ENV
        .get_or_init(|| {
            Arc::new(AppEnv {
                user_data_cache: DashMap::new(),
                vector_store_map: DashMap::new(),
            })
        })
        .clone()
}
