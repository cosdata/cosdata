use dashmap::DashMap;
use std::sync::Arc;

pub type NumericValue = Vec<f32>;
pub type VectorHash = Vec<u8>;

type CacheType = DashMap<(i8, VectorHash), Option<(VectorTreeNode, Arc<()>)>>;

#[derive(Debug, Clone)]
pub struct VectorStore {
    pub cache: Arc<CacheType>,
    pub max_cache_level: i8,
    pub database_name: String,
    pub root_vec: (VectorHash, NumericValue),
}

#[derive(Debug, Clone)]
pub struct VectorEmbedding {
    pub raw_vec: NumericValue,
    pub hash_vec: VectorHash,
}

#[derive(Debug, Clone)]
pub struct VectorTreeNode {
    pub vector_list: NumericValue,
    pub neighbors: Vec<(VectorHash, f32)>,
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
