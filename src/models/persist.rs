use rocksdb::{Options, DB};

use super::common::WaCustomError;

pub struct Persist {
    db: DB,
    cf_handles: Vec<String>,
}

impl Persist {
    pub fn new(path: &str) -> Result<Self, WaCustomError> {
        // Open the RocksDB database
        let mut options = Options::default();
        options.create_if_missing(true);
        let result = DB::open(&options, path);

        match result {
            Ok(db) => {
                // Initialize column families map with "main" entry
                let cf_handles = Vec::new();

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
    pub fn create_cf_family(&mut self, cf_name: &str) -> Result<(), WaCustomError> {
        let cf_opts = Options::default();
        let result = self.db.create_cf(&cf_name, &cf_opts);
        match result {
            Ok(_) => {
                self.cf_handles.push(cf_name.to_string());
                return Ok(());
            }
            Err(e) => Err(WaCustomError::CreateCFFailed(e.into_string())),
        }
    }

    // Put a key-value pair into a column family
    pub fn put_cf(&self, cf_name: &str, key: &[u8], value: &[u8]) -> Result<(), WaCustomError> {
        match self.db.cf_handle(cf_name) {
            Some(cf_handle) => {
                let result = self.db.put_cf(cf_handle, key, value);
                match result {
                    Ok(_) => return Ok(()),
                    Err(e) => Err(WaCustomError::CFReadWriteFailed(e.into_string())),
                }
            }
            None => Err(WaCustomError::CFNotFound),
        }
    }

    // Get a value from a column family by key
    pub fn get_cf(&self, cf_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>, WaCustomError> {
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
