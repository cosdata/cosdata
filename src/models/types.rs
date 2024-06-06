use dashmap::DashMap;
use rocksdb::{ColumnFamily, Error, Options, DB};
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
type ColumnFamilyMap = DashMap<String, ColumnFamily>;

// Define the AppEnv struct
pub struct AppEnv {
    pub user_data_cache: UserDataCache,
    pub vector_store_map: VectorStoreMap,
    pub persist: Option<Persist>,
}
// Initialize the AppEnv with once_cell
use once_cell::sync::OnceCell;

use super::common::WaCustomError;
static AIN_ENV: OnceCell<Arc<AppEnv>> = OnceCell::new();

pub fn get_app_env() -> Arc<AppEnv> {
    AIN_ENV
        .get_or_init(|| {
            let path = "./xdb/"; // Change this to your desired path
            let result = Persist::new(path);
            match result {
                Ok(value) => Arc::new(AppEnv {
                    user_data_cache: DashMap::new(),
                    vector_store_map: DashMap::new(),
                    persist: Some(value),
                }),
                Err(e) => Arc::new(AppEnv {
                    user_data_cache: DashMap::new(),
                    vector_store_map: DashMap::new(),
                    persist: None,
                }),
            }
        })
        .clone()
}

pub struct Persist {
    db: DB,
    cf_handles: Vec<String>,
}

impl Persist {
    fn new(path: &str) -> Result<Self, WaCustomError> {
        // Open the RocksDB database
        let mut options = Options::default();
        options.create_if_missing(true);
        let result = DB::open(&options, path);

        match result {
            Ok(mut db) => {
                // Initialize column families map with "main" entry
                let mut cf_handles = Vec::new();
                let cf_name = "main";
                let cf_opts = Options::default();
                db.create_cf(cf_name, &cf_opts);
                cf_handles.push(cf_name.to_string());
                Ok(Self { db, cf_handles })
            }
            Err(e) => Err(WaCustomError::CreateDatabaseFailed(e.into_string())),
        }
    }

    // Getter method for database handle
    fn get_db(&self) -> &DB {
        &self.db
    }

    // Create a new column family
    fn create_cf_family(&mut self, cf_name: &str) -> Result<(), WaCustomError> {
        let cf_opts = Options::default();
        let result = self.db.create_cf(&cf_name, &cf_opts);
        match result {
            Ok(_cf) => {
                self.cf_handles.push(cf_name.to_string());
                return Ok(());
            }
            Err(e) => Err(WaCustomError::CreateCFFailed(e.into_string())),
        }
    }

    // Put a key-value pair into a column family
    fn put_cf(&self, cf_name: &str, key: &[u8], value: &[u8]) -> Result<(), WaCustomError> {
        match self.db.cf_handle(cf_name) {
            Some(cf_handle) => {
                let result = self.db.put_cf(cf_handle, key, value);
                match result {
                    Ok(res) => return Ok(()),
                    Err(e) => Err(WaCustomError::CFReadWriteFailed(e.into_string())),
                }
            }
            None => Err(WaCustomError::CFNotFound),
        }
    }

    // Get a value from a column family by key
    fn get_cf(&self, cf_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>, WaCustomError> {
        match self.db.cf_handle(cf_name) {
            Some(cf_handle) => {
                let result = self.db.get_cf(cf_handle, key);
                match result {
                    Ok(res) => return Ok(res),
                    Err(e) => Err(WaCustomError::CFReadWriteFailed(e.into_string())),
                }
            }
            None => Err(WaCustomError::CFNotFound),
        }
    }
}
