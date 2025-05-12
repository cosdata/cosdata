use std::sync::Arc;

use crate::args::CosdataArgs;
use crate::config_loader::Config;
use crate::models::collection_cache::{CollectionCacheManager, CollectionCacheExt};
use crate::models::common::WaCustomError;
use crate::models::crypto::DoubleSHA256Hash;
use crate::models::paths::get_data_path;
use crate::models::types::{get_app_env, AppEnv};
use crate::rbac::store::RbacStore;
use crate::rbac::{Permission, RbacUser};
use rayon::ThreadPool;
use std::fmt::Write;

#[allow(unused)]
pub struct AppContext {
    pub config: Config,
    pub threadpool: ThreadPool,
    pub ain_env: Arc<AppEnv>,
    pub collection_cache_manager: Arc<CollectionCacheManager>,
    pub rbac_store: Arc<RbacStore>,
}

impl AppContext {
    pub fn new(config: Config, args: CosdataArgs) -> Result<Self, WaCustomError> {
        let ain_env = get_app_env(&config, args.clone())?;
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_pool.pool_size)
            .build()
            .expect("Failed to build thread pool");

        let collections_path = get_data_path().join("collections");
        std::fs::create_dir_all(&collections_path)
            .map_err(|e| WaCustomError::FsError(e.to_string()))?;

        let collection_cache_manager = Arc::new(CollectionCacheManager::new(
            Arc::from(collections_path),
            config.cache.max_collections,
            config.cache.eviction_probability,
            ain_env.clone(),
        ));

        let rbac_store = Arc::new(RbacStore::new(ain_env.persist.clone()).map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to initialize RBAC store: {}", e))
        })?);

        // Pass admin_key from args to setup function
        setup_initial_admin_user(&rbac_store, &args.admin_key)?;

        Ok(Self {
            config,
            ain_env,
            threadpool,
            collection_cache_manager,
            rbac_store,
        })
    }
}

fn setup_initial_admin_user(store: &RbacStore, admin_password: &str) -> Result<(), WaCustomError> {
    let admin_username = "admin";
    let admin_role_name = "admin";
    let system_collection_name = "_system";

    match store.get_user_by_username(admin_username) {
        Ok(Some(_user)) => {
            log::info!(
                "RBAC admin user '{}' already exists. Verifying role assignment...",
                admin_username
            );
            match store.check_permission(
                admin_username,
                system_collection_name,
                Permission::ManagePermissions,
            ) {
                Ok(true) => {
                    log::debug!(
                        "RBAC admin user '{}' already has necessary permissions on '{}'.",
                        admin_username,
                        system_collection_name
                    );
                    Ok(())
                }
                Ok(false) => {
                    log::warn!("RBAC admin user '{}' exists but lacks admin role assignment on '{}'. Attempting assignment.", admin_username, system_collection_name);
                    match store.assign_role_to_user(
                        admin_username,
                        system_collection_name,
                        admin_role_name,
                    ) {
                        Ok(()) => {
                            log::info!("Successfully assigned default admin role to existing admin user '{}'.", admin_username);
                            Ok(())
                        }
                        Err(e) => {
                            log::error!("Failed to assign default admin role to existing admin user '{}': {}", admin_username, e);
                            Err(WaCustomError::DatabaseError(format!(
                                "Failed to assign admin role to existing admin: {}",
                                e
                            )))
                        }
                    }
                }
                Err(e) => {
                    log::error!(
                        "Error checking permissions for existing admin user '{}': {}",
                        admin_username,
                        e
                    );
                    Err(WaCustomError::DatabaseError(format!(
                        "Failed to check admin permissions: {}",
                        e
                    )))
                }
            }
        }
        Ok(None) => {
            log::info!(
                "RBAC admin user '{}' not found, creating...",
                admin_username
            );

            let hash = DoubleSHA256Hash::new(admin_password.as_bytes());
            let mut password_hash_string = String::with_capacity(hash.0.len() * 2);
            for byte in hash.0.iter() {
                write!(&mut password_hash_string, "{:02x}", byte).unwrap();
            }

            let admin_rbac_user = RbacUser {
                user_id: 0,
                username: admin_username.to_string(),
                password_hash: password_hash_string,
                collection_roles: Vec::new(),
            };

            match store.create_user(admin_rbac_user) {
                Ok(created_user) => {
                    log::debug!(
                        "Successfully created RBAC admin user '{}' with ID {}.",
                        created_user.username,
                        created_user.user_id
                    );
                    match store.assign_role_to_user(
                        admin_username,
                        system_collection_name,
                        admin_role_name,
                    ) {
                        Ok(()) => {
                            log::debug!("Successfully assigned '{}' role to new user '{}' for collection '{}'.", admin_role_name, admin_username, system_collection_name);
                            Ok(())
                        }
                        Err(e) => {
                            log::error!("User created, but failed to assign default admin role to '{}' for '{}': {}", admin_username, system_collection_name, e);
                            Err(WaCustomError::DatabaseError(format!(
                                "User created, but failed to assign admin role: {}",
                                e
                            )))
                        }
                    }
                }
                Err(e) => {
                    log::error!(
                        "Failed to create RBAC admin user '{}': {}",
                        admin_username,
                        e
                    );
                    Err(WaCustomError::DatabaseError(format!(
                        "Failed to create RBAC admin user: {}",
                        e
                    )))
                }
            }
        }
        Err(e) => {
            log::error!(
                "Error checking for RBAC admin user '{}': {}",
                admin_username,
                e
            );
            Err(WaCustomError::DatabaseError(format!(
                "Failed to check RBAC admin user: {}",
                e
            )))
        }
    }
}


impl CollectionCacheExt for AppContext {
    fn update_collection_for_transaction(&self, name: &str) -> Result<(), WaCustomError> {
        self.collection_cache_manager.update_collection_usage(name)
    }

    fn update_collection_for_query(&self, name: &str) -> Result<bool, WaCustomError> {
        self.collection_cache_manager
            .probabilistic_update(name, 0.01)
    }
}
