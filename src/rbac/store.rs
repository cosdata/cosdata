use crate::models::common::WaCustomError;
use crate::rbac::{Permission, RbacCollection, RbacUser, Role};
use lmdb::{Cursor, Database, Environment, Transaction, WriteFlags};
use std::sync::Arc;

const USER_DB_NAME: &str = "rbac_users";
const ROLE_DB_NAME: &str = "rbac_roles";
const COLLECTION_DB_NAME: &str = "rbac_collections";

const USER_COUNTER_KEY: &[u8] = b"id_counter:user";
const ROLE_COUNTER_KEY: &[u8] = b"id_counter:role";
const COLLECTION_COUNTER_KEY: &[u8] = b"id_counter:collection";

pub struct RbacStore {
    env: Arc<Environment>,
    user_db: Database,
    role_db: Database,
    collection_db: Database,
}

fn entity_key(prefix: &str, id: u32) -> String {
    format!("{}:{}", prefix, id)
}

impl RbacStore {
    pub fn new(env: Arc<Environment>) -> Result<Self, WaCustomError> {
        let user_db = env
            .create_db(Some(USER_DB_NAME), lmdb::DatabaseFlags::empty())
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to create user DB: {}", e))
            })?;

        let role_db = env
            .create_db(Some(ROLE_DB_NAME), lmdb::DatabaseFlags::empty())
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to create role DB: {}", e))
            })?;

        let collection_db = env
            .create_db(Some(COLLECTION_DB_NAME), lmdb::DatabaseFlags::empty())
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to create collection DB: {}", e))
            })?;

        let store = Self {
            env,
            user_db,
            role_db,
            collection_db,
        };

        if store.is_roles_db_empty()? {
            store.init_default_roles()?;
        }

        Ok(store)
    }

    fn is_roles_db_empty(&self) -> Result<bool, WaCustomError> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let mut cursor = txn
            .open_ro_cursor(self.role_db)
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let is_empty = cursor.iter().next().is_none();
        Ok(is_empty)
    }

    fn init_default_roles(&self) -> Result<(), WaCustomError> {
        use crate::rbac::{get_admin_role, get_reader_role, get_writer_role};

        let admin_role = get_admin_role();
        let reader_role = get_reader_role();
        let writer_role = get_writer_role();

        self.create_role(admin_role)?;
        self.create_role(reader_role)?;
        self.create_role(writer_role)?;

        let system_collection_name = "_system";
        match self.get_collection_by_name(system_collection_name) {
            Ok(Some(_)) => { /* Already exists */ }
            Ok(None) => {
                log::info!(
                    "Creating default RBAC collection entry '{}'",
                    system_collection_name
                );
                let system_collection = RbacCollection {
                    collection_id: 0,
                    collection_name: system_collection_name.to_string(),
                };
                self.create_collection(system_collection)?;
            }
            Err(e) => {
                log::error!(
                    "Failed to check/create default RBAC collection entry: {}",
                    e
                );
                return Err(e);
            }
        }

        Ok(())
    }

    // Generate a new ID for the given entity type
    fn generate_id(&self, db: Database, counter_key: &[u8]) -> Result<u32, WaCustomError> {
        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RW txn for generate_id: {}", e))
        })?;

        let key_vec = counter_key.to_vec();

        let current_id = match txn.get(db, &key_vec) {
            Ok(bytes) => bytes.try_into().map(u32::from_le_bytes).unwrap_or(0),
            Err(lmdb::Error::NotFound) => 0,
            Err(e) => {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get counter key '{:?}': {}",
                    String::from_utf8_lossy(counter_key),
                    e
                )));
            }
        };

        let next_id = current_id.checked_add(1).ok_or_else(|| {
            let err_msg = format!(
                "ID counter overflow for key '{:?}'",
                String::from_utf8_lossy(counter_key)
            );
            log::error!("{}", err_msg);
            WaCustomError::DatabaseError(err_msg)
        })?;

        let id_bytes = next_id.to_le_bytes();

        txn.put(db, &key_vec, &id_bytes, WriteFlags::empty())
            .map_err(|e| {
                WaCustomError::DatabaseError(format!(
                    "Failed to put counter key '{:?}': {}",
                    String::from_utf8_lossy(counter_key),
                    e
                ))
            })?;

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit generate_id txn: {}", e))
        })?;

        Ok(next_id)
    }

    // User CRUD operations

    pub fn create_user(&self, mut user: RbacUser) -> Result<RbacUser, WaCustomError> {
        if user.user_id == 0 {
            user.user_id = self.generate_id(self.user_db, USER_COUNTER_KEY)?;
        }

        let user_id_bytes = user.user_id.to_le_bytes();
        let user_bytes = serde_cbor::to_vec(&user)
            .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

        let mut txn = self
            .env
            .begin_rw_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let user_key = entity_key("user", user.user_id);
        txn.put(self.user_db, &user_key, &user_bytes, WriteFlags::empty())
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let username_bytes = user.username.as_bytes().to_vec();
        txn.put(
            self.user_db,
            &username_bytes,
            &user_id_bytes,
            WriteFlags::empty(),
        )
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.commit()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        Ok(user)
    }

    pub fn get_user_by_id(&self, user_id: u32) -> Result<Option<RbacUser>, WaCustomError> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let user_key = entity_key("user", user_id);
        let result = match txn.get(self.user_db, &user_key) {
            Ok(bytes) => {
                let user: RbacUser = serde_cbor::from_slice(bytes)
                    .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;
                Some(user)
            }
            Err(lmdb::Error::NotFound) => None,
            Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
        };

        Ok(result)
    }

    pub fn get_user_by_username(&self, username: &str) -> Result<Option<RbacUser>, WaCustomError> {
        let txn = self.env.begin_ro_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to start RO txn for get_user_by_username: {}",
                e
            ))
        })?;

        let username_bytes = username.as_bytes().to_vec();

        let user_id_bytes = match txn.get(self.user_db, &username_bytes) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => return Ok(None),
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get user mapping: {}",
                    e
                )))
            }
        };

        let user_id = user_id_bytes
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| {
                WaCustomError::DeserializationError(
                    "Invalid user ID format found in DB".to_string(),
                )
            })?;

        let user_key = entity_key("user", user_id);
        let result = match txn.get(self.user_db, &user_key) {
            Ok(bytes) => {
                // Deserialize the user data
                serde_cbor::from_slice(bytes).map_err(|e| {
                    WaCustomError::DeserializationError(format!(
                        "Failed to deserialize user data: {}",
                        e
                    ))
                })?
            }
            Err(lmdb::Error::NotFound) => {
                log::error!("Inconsistent RBAC state: User ID mapping found for '{}', but user record missing.", username);
                return Err(WaCustomError::DatabaseError(format!(
                    "Inconsistent state for user '{}'",
                    username
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get user record: {}",
                    e
                )))
            }
        };
        Ok(Some(result))
    }

    // Role CRUD operations

    pub fn create_role(&self, mut role: Role) -> Result<Role, WaCustomError> {
        if role.role_id == 0 {
            role.role_id = self.generate_id(self.role_db, ROLE_COUNTER_KEY)?;
            log::debug!(
                "Generated new role_id {} for role '{}'",
                role.role_id,
                role.role_name
            );
        } else {
            log::debug!(
                "Role '{}' provided with hardcoded role_id {}. Ensuring counter is advanced.",
                role.role_name,
                role.role_id
            );
            if let Err(e) =
                self.ensure_counter_advanced(self.role_db, ROLE_COUNTER_KEY, role.role_id)
            {
                log::error!(
                    "Failed to advance role counter past {}: {}",
                    role.role_id,
                    e
                );
                return Err(e);
            }
        }

        let role_id_bytes = role.role_id.to_le_bytes();
        let role_bytes = match serde_cbor::to_vec(&role) {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(WaCustomError::SerializationError(format!(
                    "Failed to serialize role '{}': {}",
                    role.role_name, e
                )))
            }
        };

        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RW txn for create_role: {}", e))
        })?;

        let role_key = entity_key("role", role.role_id);
        if let Err(e) = txn.put(
            self.role_db,
            &role_key,
            &role_bytes,
            WriteFlags::NO_OVERWRITE,
        ) {
            if e == lmdb::Error::KeyExist {
                txn.abort();
                log::error!(
                    "Attempted to create role with ID {}, but it already exists.",
                    role.role_id
                );
                return Err(WaCustomError::DatabaseError(format!(
                    "Role ID {} already exists",
                    role.role_id
                )));
            } else {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to put role data: {}",
                    e
                )));
            }
        }

        let role_name_bytes = role.role_name.as_bytes().to_vec();
        if let Err(e) = txn.put(
            self.role_db,
            &role_name_bytes,
            &role_id_bytes,
            WriteFlags::NO_OVERWRITE,
        ) {
            if e == lmdb::Error::KeyExist {
                txn.abort();
                log::error!(
                    "Attempted to create role with name '{}', but it already exists.",
                    role.role_name
                );
                return Err(WaCustomError::DatabaseError(format!(
                    "Role name '{}' already exists",
                    role.role_name
                )));
            } else {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to put role name mapping: {}",
                    e
                )));
            }
        }

        // Commit transaction
        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit create_role txn: {}", e))
        })?;

        log::debug!(
            "Successfully created role '{}' with ID {}",
            role.role_name,
            role.role_id
        );
        Ok(role)
    }

    fn ensure_counter_advanced(
        &self,
        db: Database,
        counter_key: &[u8],
        minimum_id: u32,
    ) -> Result<(), WaCustomError> {
        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to start RW txn for ensure_counter: {}",
                e
            ))
        })?;

        let key_vec = counter_key.to_vec();

        // Get current counter value
        let current_id = match txn.get(db, &key_vec) {
            Ok(bytes) => bytes.try_into().map(u32::from_le_bytes).unwrap_or(0),
            Err(lmdb::Error::NotFound) => 0,
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed get counter '{:?}': {}",
                    String::from_utf8_lossy(counter_key),
                    e
                )));
            }
        };

        if current_id < minimum_id {
            let id_bytes = minimum_id.to_le_bytes();
            txn.put(db, &key_vec, &id_bytes, WriteFlags::empty())
                .map_err(|e| {
                    WaCustomError::DatabaseError(format!(
                        "Failed put counter '{:?}': {}",
                        String::from_utf8_lossy(counter_key),
                        e
                    ))
                })?;
            txn.commit().map_err(|e| {
                WaCustomError::DatabaseError(format!(
                    "Failed commit counter '{:?}': {}",
                    String::from_utf8_lossy(counter_key),
                    e
                ))
            })?;
        } else {
            txn.abort();
            log::debug!(
                "Counter '{:?}' is already at or above {}. Current: {}",
                String::from_utf8_lossy(counter_key),
                minimum_id,
                current_id
            );
        }
        Ok(())
    }

    pub fn get_role_by_id(&self, role_id: u32) -> Result<Option<Role>, WaCustomError> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let role_key = entity_key("role", role_id);
        let result = match txn.get(self.role_db, &role_key) {
            Ok(bytes) => {
                let role: Role = serde_cbor::from_slice(bytes)
                    .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;
                Some(role)
            }
            Err(lmdb::Error::NotFound) => None,
            Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
        };

        Ok(result)
    }

    pub fn get_role_by_name(&self, role_name: &str) -> Result<Option<Role>, WaCustomError> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let role_name_bytes = role_name.as_bytes().to_vec();
        let role_id_bytes = match txn.get(self.role_db, &role_name_bytes) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => return Ok(None),
            Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
        };

        if role_id_bytes.len() != 4 {
            return Err(WaCustomError::DeserializationError(
                "Invalid role ID format".to_string(),
            ));
        }

        let role_id = u32::from_le_bytes([
            role_id_bytes[0],
            role_id_bytes[1],
            role_id_bytes[2],
            role_id_bytes[3],
        ]);

        self.get_role_by_id(role_id)
    }

    // Collection CRUD operations

    pub fn create_collection(
        &self,
        mut collection: RbacCollection,
    ) -> Result<RbacCollection, WaCustomError> {
        if collection.collection_id == 0 {
            collection.collection_id =
                self.generate_id(self.collection_db, COLLECTION_COUNTER_KEY)?;
        }

        let collection_id_bytes = collection.collection_id.to_le_bytes();
        let collection_bytes = serde_cbor::to_vec(&collection)
            .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

        let mut txn = self
            .env
            .begin_rw_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collection_key = entity_key("collection", collection.collection_id);
        txn.put(
            self.collection_db,
            &collection_key,
            &collection_bytes,
            WriteFlags::empty(),
        )
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collection_name_bytes = collection.collection_name.as_bytes().to_vec();
        txn.put(
            self.collection_db,
            &collection_name_bytes,
            &collection_id_bytes,
            WriteFlags::empty(),
        )
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.commit()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        Ok(collection)
    }

    pub fn get_collection_by_name(
        &self,
        collection_name: &str,
    ) -> Result<Option<RbacCollection>, WaCustomError> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collection_name_bytes = collection_name.as_bytes().to_vec();
        let collection_id_bytes = match txn.get(self.collection_db, &collection_name_bytes) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => return Ok(None),
            Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
        };

        if collection_id_bytes.len() != 4 {
            return Err(WaCustomError::DeserializationError(
                "Invalid collection ID format".to_string(),
            ));
        }

        let collection_id = u32::from_le_bytes([
            collection_id_bytes[0],
            collection_id_bytes[1],
            collection_id_bytes[2],
            collection_id_bytes[3],
        ]);

        self.get_collection_by_id(collection_id)
    }

    pub fn get_collection_by_id(
        &self,
        collection_id: u32,
    ) -> Result<Option<RbacCollection>, WaCustomError> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let collection_key = entity_key("collection", collection_id);
        let result = match txn.get(self.collection_db, &collection_key) {
            Ok(bytes) => {
                let collection: RbacCollection = serde_cbor::from_slice(bytes)
                    .map_err(|e| WaCustomError::DeserializationError(e.to_string()))?;
                Some(collection)
            }
            Err(lmdb::Error::NotFound) => None,
            Err(e) => return Err(WaCustomError::DatabaseError(e.to_string())),
        };

        Ok(result)
    }

    /// Gets an RbacCollection by its ID or name
    #[allow(unused_assignments)]
    pub fn get_collection_by_id_or_name(
        &self,
        id_or_name: &str,
    ) -> Result<Option<RbacCollection>, WaCustomError> {
        let txn = self.env.begin_ro_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed start RO txn get_coll_by_id_or_name: {}",
                e
            ))
        })?;

        let collection_key_or_name_bytes = id_or_name.as_bytes().to_vec();
        let mut collection_id_to_lookup: Option<u32> = None;
        let mut collection_key_for_lookup: Option<String> = None;

        if let Ok(id) = id_or_name.parse::<u32>() {
            collection_id_to_lookup = Some(id);
            collection_key_for_lookup = Some(entity_key("collection", id));
        } else {
            match txn.get(self.collection_db, &collection_key_or_name_bytes) {
                Ok(id_bytes) => match id_bytes.try_into().map(u32::from_le_bytes) {
                    Ok(id) => {
                        collection_id_to_lookup = Some(id);
                        collection_key_for_lookup = Some(entity_key("collection", id));
                    }
                    Err(_) => {
                        return Err(WaCustomError::DeserializationError(
                            "Invalid collection ID format in DB".to_string(),
                        ))
                    }
                },
                Err(lmdb::Error::NotFound) => return Ok(None),
                Err(e) => {
                    return Err(WaCustomError::DatabaseError(format!(
                        "Failed get coll name mapping: {}",
                        e
                    )))
                }
            }
        }

        if let Some(coll_key) = collection_key_for_lookup {
            match txn.get(self.collection_db, &coll_key) {
                Ok(bytes) => match serde_cbor::from_slice(bytes) {
                    Ok(collection) => Ok(Some(collection)),
                    Err(e) => Err(WaCustomError::DeserializationError(format!(
                        "Failed deserialize coll data: {}",
                        e
                    ))),
                },
                Err(lmdb::Error::NotFound) => {
                    log::error!("Inconsistent RBAC state: Collection ID {} found but record missing for '{}'.",
                                 collection_id_to_lookup.unwrap_or(0), id_or_name);
                    Ok(None)
                }
                Err(e) => Err(WaCustomError::DatabaseError(format!(
                    "Failed get coll record: {}",
                    e
                ))),
            }
        } else {
            Ok(None)
        }
    }

    pub fn check_permission(
        &self,
        username: &str,
        collection_name: &str,
        required_permission: Permission,
    ) -> Result<bool, WaCustomError> {
        let txn = self.env.begin_ro_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to start RO txn for check_permission: {}",
                e
            ))
        })?;

        let username_bytes = username.as_bytes().to_vec();
        let user_id_bytes = match txn.get(self.user_db, &username_bytes) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => {
                log::debug!(
                    "check_permission: User mapping not found for '{}'",
                    username
                );
                return Ok(false);
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get user mapping for '{}': {}",
                    username, e
                )))
            }
        };
        let user_id = match user_id_bytes.try_into().map(u32::from_le_bytes) {
            Ok(id) => id,
            Err(_) => {
                return Err(WaCustomError::DeserializationError(format!(
                    "Invalid user ID format for user '{}'",
                    username
                )))
            }
        };

        let user_key = entity_key("user", user_id);
        let user: RbacUser = match txn.get(self.user_db, &user_key) {
            Ok(bytes) => match serde_cbor::from_slice(bytes) {
                Ok(u) => u,
                Err(e) => {
                    return Err(WaCustomError::DeserializationError(format!(
                        "Failed to deserialize user data for ID {}: {}",
                        user_id, e
                    )))
                }
            },
            Err(lmdb::Error::NotFound) => {
                log::error!(
                    "check_permission: Inconsistent state: User record missing for ID {}.",
                    user_id
                );
                return Ok(false);
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get user record for ID {}: {}",
                    user_id, e
                )))
            }
        };

        let collection_name_bytes = collection_name.as_bytes().to_vec();
        let collection_id_bytes = match txn.get(self.collection_db, &collection_name_bytes) {
            Ok(bytes) => bytes,
            Err(lmdb::Error::NotFound) => {
                log::debug!(
                    "check_permission: Collection mapping not found for name '{}'",
                    collection_name
                );
                return Ok(false);
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get collection mapping for name '{}': {}",
                    collection_name, e
                )))
            }
        };
        let collection_id = match collection_id_bytes.try_into().map(u32::from_le_bytes) {
            Ok(id) => id,
            Err(_) => {
                return Err(WaCustomError::DeserializationError(format!(
                    "Invalid collection ID format for name '{}'",
                    collection_name
                )))
            }
        };

        let target_role_ids: Vec<u32> = user
            .collection_roles
            .iter()
            .filter(|(col_id, _)| *col_id == collection_id)
            .map(|(_, role_id)| *role_id)
            .collect();

        if target_role_ids.is_empty() {
            log::info!(
                "check_permission: No roles assigned for user '{}' on collection '{}' (ID {}).",
                username,
                collection_name,
                collection_id
            );
            return Ok(false);
        }

        for role_id in target_role_ids {
            let role_key = entity_key("role", role_id);
            match txn.get(self.role_db, &role_key) {
                Ok(role_bytes) => match serde_cbor::from_slice::<Role>(role_bytes) {
                    Ok(role) => {
                        if role.permissions.contains(&required_permission) {
                            return Ok(true);
                        } else {
                            log::info!(
                                "check_permission: Permission NOT FOUND in role '{}'",
                                role.role_name
                            );
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "check_permission: Failed to deserialize role data for ID {}: {}",
                            role_id,
                            e
                        );
                        continue;
                    }
                },
                Err(lmdb::Error::NotFound) => {
                    log::warn!("check_permission: Role ID {} (assigned to user '{}' for collection '{}') not found in role DB.", role_id, username, collection_name);
                    continue;
                }
                Err(e) => {
                    // Database error fetching role
                    return Err(WaCustomError::DatabaseError(format!(
                        "Failed to get role record for ID {}: {}",
                        role_id, e
                    )));
                }
            }
        }

        log::info!("--- Exit check_permission: Permission '{:?}' not found for user '{}' on collection '{}' ---", required_permission, username, collection_name);
        Ok(false)
    }

    pub fn assign_role_to_user(
        &self,
        username: &str,
        collection_name: &str,
        role_name: &str,
    ) -> Result<(), WaCustomError> {
        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RW txn for assign_role: {}", e))
        })?;

        let username_bytes = username.as_bytes().to_vec();
        let user_id_bytes = match txn.get(self.user_db, &username_bytes) {
            Ok(bytes) => bytes.to_vec(),
            Err(lmdb::Error::NotFound) => {
                return Err(WaCustomError::NotFound(format!(
                    "User '{}' not found",
                    username
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get user mapping: {}",
                    e
                )));
            }
        };
        let user_id = user_id_bytes
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| {
                WaCustomError::DeserializationError("Invalid user ID format".to_string())
            })?;

        let user_key = entity_key("user", user_id);
        let mut user: RbacUser = match txn.get(self.user_db, &user_key) {
            Ok(bytes) => serde_cbor::from_slice(bytes).map_err(|e| {
                WaCustomError::DeserializationError(format!(
                    "Failed to deserialize user data: {}",
                    e
                ))
            })?,
            Err(lmdb::Error::NotFound) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Inconsistent state: User ID mapping found for '{}', but user record missing.",
                    username
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get user record: {}",
                    e
                )));
            }
        };

        let collection_name_bytes = collection_name.as_bytes().to_vec();
        let collection_id_bytes = match txn.get(self.collection_db, &collection_name_bytes) {
            Ok(bytes) => bytes.to_vec(),
            Err(lmdb::Error::NotFound) => {
                return Err(WaCustomError::NotFound(format!(
                    "Collection '{}' not found",
                    collection_name
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get collection mapping: {}",
                    e
                )));
            }
        };
        let collection_id = collection_id_bytes
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| {
                WaCustomError::DeserializationError("Invalid collection ID format".to_string())
            })?;

        let role_name_bytes = role_name.as_bytes().to_vec();
        let role_id_bytes = match txn.get(self.role_db, &role_name_bytes) {
            Ok(bytes) => bytes.to_vec(),
            Err(lmdb::Error::NotFound) => {
                return Err(WaCustomError::NotFound(format!(
                    "Role '{}' not found",
                    role_name
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get role mapping: {}",
                    e
                )));
            }
        };
        let role_id = role_id_bytes
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| {
                WaCustomError::DeserializationError("Invalid role ID format".to_string())
            })?;

        let role_pair_to_add = (collection_id, role_id);
        user.collection_roles
            .retain(|(col_id, _)| *col_id != collection_id);
        if !user.collection_roles.contains(&role_pair_to_add) {
            user.collection_roles.push(role_pair_to_add);
        }

        let updated_user_bytes = match serde_cbor::to_vec(&user) {
            Ok(bytes) => bytes,
            Err(e) => {
                return Err(WaCustomError::SerializationError(format!(
                    "Failed to serialize updated user data: {}",
                    e
                )));
            }
        };

        txn.put(
            self.user_db,
            &user_key,
            &updated_user_bytes,
            WriteFlags::empty(),
        )
        .map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to put updated user data during role assignment: {}",
                e
            ))
        })?;

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit role assignment: {}", e))
        })?;

        Ok(())
    }

    pub fn get_all_users(&self) -> Result<Vec<RbacUser>, WaCustomError> {
        let txn = self
            .env
            .begin_ro_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let mut cursor = txn
            .open_ro_cursor(self.user_db)
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        let mut users = Vec::new();

        for (key_bytes, value_bytes) in cursor.iter() {
            let key = String::from_utf8_lossy(key_bytes);

            if key.starts_with("user:") {
                match serde_cbor::from_slice::<RbacUser>(value_bytes) {
                    Ok(user) => users.push(user),
                    Err(e) => {
                        eprintln!("Error deserializing user: {}", e);
                    }
                }
            }
        }

        Ok(users)
    }

    pub fn delete_user_by_username(&self, username: &str) -> Result<(), WaCustomError> {
        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to start RW transaction for delete: {}",
                e
            ))
        })?;

        let username_bytes = username.as_bytes().to_vec();

        let user_id_bytes = match txn.get(self.user_db, &username_bytes) {
            Ok(bytes) => bytes.to_vec(),
            Err(lmdb::Error::NotFound) => {
                txn.abort();
                return Err(WaCustomError::NotFound(format!(
                    "User '{}' not found",
                    username
                )));
            }
            Err(e) => {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get username mapping: {}",
                    e
                )));
            }
        };

        let user_id = user_id_bytes
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| {
                WaCustomError::DeserializationError(
                    "Invalid user ID format found in DB".to_string(),
                )
            })?;

        let user_key = entity_key("user", user_id);

        match txn.del(self.user_db, &user_key, None) {
            Ok(_) => (),
            Err(lmdb::Error::NotFound) => {
                log::warn!("User record key '{}' not found during deletion by username '{}', but proceeding.", user_key, username);
            }
            Err(e) => {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to delete primary user record: {}",
                    e
                )));
            }
        }

        match txn.del(self.user_db, &username_bytes, None) {
            Ok(_) => (),
            Err(lmdb::Error::NotFound) => {
                log::warn!(
                    "Username mapping key '{}' not found during deletion, but proceeding.",
                    username
                );
            }
            Err(e) => {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to delete username mapping: {}",
                    e
                )));
            }
        }

        // Commit the transaction
        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit delete transaction: {}", e))
        })?;

        Ok(())
    }

    pub fn update_user(&self, user: RbacUser) -> Result<(), WaCustomError> {
        let user_bytes = serde_cbor::to_vec(&user)
            .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RW txn for update: {}", e))
        })?;

        let user_key = entity_key("user", user.user_id);

        // Fetch current username
        let current_username: Option<String> = match txn.get(self.user_db, &user_key) {
            Ok(bytes) => serde_cbor::from_slice::<RbacUser>(bytes)
                .ok()
                .map(|u| u.username),
            Err(_) => None,
        };

        txn.put(self.user_db, &user_key, &user_bytes, WriteFlags::empty())
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to put updated user data: {}", e))
            })?;

        let username_changed = match &current_username {
            Some(old_name) => *old_name != user.username,
            None => true,
        };

        if username_changed {
            if let Some(old_name) = &current_username {
                let old_username_bytes = old_name.as_bytes().to_vec();
                match txn.del(self.user_db, &old_username_bytes, None) {
                    Ok(_) | Err(lmdb::Error::NotFound) => (),
                    Err(e) => {
                        txn.abort();
                        return Err(WaCustomError::DatabaseError(format!(
                            "Failed to delete old username mapping: {}",
                            e
                        )));
                    }
                }
            }

            let new_username_bytes = user.username.as_bytes().to_vec();
            txn.put(
                self.user_db,
                &new_username_bytes,
                &user.user_id.to_le_bytes(),
                WriteFlags::empty(),
            )
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to put new username mapping: {}", e))
            })?;
        }

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit user update: {}", e))
        })?;

        Ok(())
    }

    pub fn get_all_roles(&self) -> Result<Vec<Role>, WaCustomError> {
        let txn = self.env.begin_ro_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RO txn for get_all_roles: {}", e))
        })?;

        let mut cursor = txn.open_ro_cursor(self.role_db).map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to open cursor for roles DB: {}", e))
        })?;

        let mut roles = Vec::new();

        for (key_bytes, value_bytes) in cursor.iter() {
            let key = String::from_utf8_lossy(key_bytes);

            if key.starts_with("role:") {
                match serde_cbor::from_slice::<Role>(value_bytes) {
                    Ok(role) => roles.push(role),
                    Err(e) => {
                        log::error!("Error deserializing role with key '{}': {}", key, e);
                    }
                }
            }
        }

        Ok(roles)
    }

    pub fn update_role(&self, role: Role) -> Result<(), WaCustomError> {
        let role_bytes = serde_cbor::to_vec(&role)
            .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RW txn for update_role: {}", e))
        })?;

        let role_key = entity_key("role", role.role_id);

        let current_role_name: Option<String> = match txn.get(self.role_db, &role_key) {
            Ok(bytes) => serde_cbor::from_slice::<Role>(bytes)
                .ok()
                .map(|r| r.role_name),
            Err(lmdb::Error::NotFound) => {
                txn.abort();
                return Err(WaCustomError::NotFound(format!(
                    "Role with ID {} not found for update",
                    role.role_id
                )));
            }
            Err(e) => {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get current role data: {}",
                    e
                )));
            }
        };

        txn.put(self.role_db, &role_key, &role_bytes, WriteFlags::empty())
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to put updated role data: {}", e))
            })?;

        let role_name_changed = match &current_role_name {
            Some(old_name) => *old_name != role.role_name,
            None => true,
        };

        if role_name_changed {
            if let Some(old_name) = &current_role_name {
                let old_name_bytes = old_name.as_bytes().to_vec();
                match txn.del(self.role_db, &old_name_bytes, None) {
                    Ok(_) | Err(lmdb::Error::NotFound) => (),
                    Err(e) => {
                        txn.abort();
                        return Err(WaCustomError::DatabaseError(format!(
                            "Failed to delete old role name mapping: {}",
                            e
                        )));
                    }
                }
            }

            let new_name_bytes = role.role_name.as_bytes().to_vec();
            txn.put(
                self.role_db,
                &new_name_bytes,
                &role.role_id.to_le_bytes(),
                WriteFlags::empty(),
            )
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed to put new role name mapping: {}", e))
            })?;
        }

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit role update: {}", e))
        })?;

        Ok(())
    }

    // ROLE METHODS

    pub fn delete_role_by_name(&self, role_name: &str) -> Result<(), WaCustomError> {
        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RW txn for delete_role: {}", e))
        })?;

        let role_name_bytes = role_name.as_bytes().to_vec();

        // Get Role ID
        let role_id = match txn.get(self.role_db, &role_name_bytes) {
            Ok(bytes) => bytes
                .to_vec()
                .as_slice()
                .try_into()
                .map(u32::from_le_bytes)
                .map_err(|_| {
                    WaCustomError::DeserializationError("Invalid role ID format".to_string())
                }),
            Err(lmdb::Error::NotFound) => {
                txn.abort();
                return Err(WaCustomError::NotFound(format!(
                    "Role '{}' not found",
                    role_name
                )));
            }
            Err(e) => {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed get role name mapping: {}",
                    e
                )));
            }
        }?;

        let role_key = entity_key("role", role_id);

        // Delete Role Records
        let del1_result = txn.del(self.role_db, &role_key, None);
        let del2_result = txn.del(self.role_db, &role_name_bytes, None);
        if let Err(e) = del1_result {
            if e != lmdb::Error::NotFound {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed delete role record: {}",
                    e
                )));
            }
        }
        if let Err(e) = del2_result {
            if e != lmdb::Error::NotFound {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed delete role name mapping: {}",
                    e
                )));
            }
        }

        log::info!(
            "Scanning users for cleanup of deleted role '{}' (ID {})",
            role_name,
            role_id
        );
        let mut users_to_update: Vec<(Vec<u8>, RbacUser)> = Vec::new();

        match txn.open_ro_cursor(self.user_db) {
            Ok(mut user_cursor) => {
                for (key_bytes, value_bytes) in user_cursor.iter_from(b"user:".as_ref()) {
                    if !key_bytes.starts_with(b"user:") {
                        break;
                    }

                    match serde_cbor::from_slice::<RbacUser>(value_bytes) {
                        Ok(mut user) => {
                            let initial_len = user.collection_roles.len();
                            user.collection_roles.retain(|(_, r_id)| *r_id != role_id);
                            if user.collection_roles.len() < initial_len {
                                users_to_update.push((key_bytes.to_vec(), user));
                            }
                        }
                        Err(e) => {
                            log::error!(
                                "Failed deserializing user {:?} during role cleanup scan: {}",
                                String::from_utf8_lossy(key_bytes),
                                e
                            );
                        }
                    }
                }
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed open user RO cursor for cleanup: {}",
                    e
                )));
            }
        }

        log::info!(
            "Applying updates to {} user records for role ID {} cleanup.",
            users_to_update.len(),
            role_id
        );
        for (user_key_bytes, updated_user) in users_to_update {
            match serde_cbor::to_vec(&updated_user) {
                Ok(updated_bytes) => {
                    if let Err(e) = txn.put(
                        self.user_db,
                        &user_key_bytes,
                        &updated_bytes,
                        WriteFlags::empty(),
                    ) {
                        log::error!(
                            "Failed writing updated user {:?} during role cleanup: {}",
                            String::from_utf8_lossy(&user_key_bytes),
                            e
                        );
                        txn.abort();
                        return Err(WaCustomError::DatabaseError(format!(
                            "Failed writing updated user: {}",
                            e
                        )));
                    }
                }
                Err(e) => {
                    log::error!(
                        "Failed serializing updated user {:?} during role cleanup: {}",
                        String::from_utf8_lossy(&user_key_bytes),
                        e
                    );
                    txn.abort();
                    return Err(WaCustomError::SerializationError(format!(
                        "Failed serializing updated user: {}",
                        e
                    )));
                }
            }
        }
        log::info!(
            "Finished user update application for role ID {} cleanup.",
            role_id
        );

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit role delete and cleanup: {}", e))
        })?;

        Ok(())
    }

    //  RBAC COLLECTION METHODS

    pub fn get_all_collections(&self) -> Result<Vec<RbacCollection>, WaCustomError> {
        let txn = self.env.begin_ro_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to start RO txn for get_all_collections: {}",
                e
            ))
        })?;

        let mut cursor = txn.open_ro_cursor(self.collection_db).map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to open cursor for collection DB: {}", e))
        })?;

        let mut collections = Vec::new();

        for (key_bytes, value_bytes) in cursor.iter() {
            let key = String::from_utf8_lossy(key_bytes);
            if key.starts_with("collection:") {
                match serde_cbor::from_slice::<RbacCollection>(value_bytes) {
                    Ok(collection) => collections.push(collection),
                    Err(e) => {
                        log::error!("Error deserializing collection with key '{}': {}", key, e);
                    }
                }
            }
        }
        Ok(collections)
    }

    pub fn update_collection(&self, collection: RbacCollection) -> Result<(), WaCustomError> {
        let collection_bytes = serde_cbor::to_vec(&collection)
            .map_err(|e| WaCustomError::SerializationError(e.to_string()))?;

        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to start RW txn for update_collection: {}",
                e
            ))
        })?;

        let collection_key = entity_key("collection", collection.collection_id);

        let current_collection_name: Option<String> =
            match txn.get(self.collection_db, &collection_key) {
                Ok(bytes) => serde_cbor::from_slice::<RbacCollection>(bytes)
                    .ok()
                    .map(|c| c.collection_name),
                Err(lmdb::Error::NotFound) => {
                    txn.abort();
                    return Err(WaCustomError::NotFound(format!(
                        "Collection with ID {} not found for update",
                        collection.collection_id
                    )));
                }
                Err(e) => {
                    txn.abort();
                    return Err(WaCustomError::DatabaseError(format!(
                        "Failed to get current collection data: {}",
                        e
                    )));
                }
            };

        txn.put(
            self.collection_db,
            &collection_key,
            &collection_bytes,
            WriteFlags::empty(),
        )
        .map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to put updated collection data: {}", e))
        })?;

        let name_changed = match &current_collection_name {
            Some(old_name) => *old_name != collection.collection_name,
            None => true,
        };

        if name_changed {
            if let Some(old_name) = current_collection_name {
                let old_name_bytes = old_name.as_bytes().to_vec();
                match txn.del(self.collection_db, &old_name_bytes, None) {
                    Ok(_) | Err(lmdb::Error::NotFound) => (),
                    Err(e) => {
                        txn.abort();
                        return Err(WaCustomError::DatabaseError(format!(
                            "Failed to delete old collection name mapping: {}",
                            e
                        )));
                    }
                }
            }
            let new_name_bytes = collection.collection_name.as_bytes().to_vec();
            txn.put(
                self.collection_db,
                &new_name_bytes,
                &collection.collection_id.to_le_bytes(),
                WriteFlags::empty(),
            )
            .map_err(|e| {
                WaCustomError::DatabaseError(format!(
                    "Failed to put new collection name mapping: {}",
                    e
                ))
            })?;
        }

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to commit collection update: {}", e))
        })?;

        Ok(())
    }

    pub fn delete_collection_by_name(&self, collection_name: &str) -> Result<(), WaCustomError> {
        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to start RW txn for delete_collection: {}",
                e
            ))
        })?;

        let name_bytes = collection_name.as_bytes().to_vec();

        //  Get Collection ID
        let collection_id = match txn.get(self.collection_db, &name_bytes) {
            Ok(bytes) => bytes
                .to_vec()
                .as_slice()
                .try_into()
                .map(u32::from_le_bytes)
                .map_err(|_| {
                    WaCustomError::DeserializationError("Invalid collection ID format".to_string())
                }),
            Err(lmdb::Error::NotFound) => {
                txn.abort();
                return Err(WaCustomError::NotFound(format!(
                    "Collection '{}' not found",
                    collection_name
                )));
            }
            Err(e) => {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed get collection name mapping: {}",
                    e
                )));
            }
        }?;

        let collection_key = entity_key("collection", collection_id);

        let del1_result = txn.del(self.collection_db, &collection_key, None);
        let del2_result = txn.del(self.collection_db, &name_bytes, None);
        if let Err(e) = del1_result {
            if e != lmdb::Error::NotFound {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed delete coll record: {}",
                    e
                )));
            }
        }
        if let Err(e) = del2_result {
            if e != lmdb::Error::NotFound {
                txn.abort();
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed delete coll name mapping: {}",
                    e
                )));
            }
        }

        log::info!(
            "Scanning users for cleanup of deleted RBAC collection '{}' (ID {})",
            collection_name,
            collection_id
        );
        let mut users_to_update: Vec<(Vec<u8>, RbacUser)> = Vec::new();

        match txn.open_ro_cursor(self.user_db) {
            Ok(mut user_cursor) => {
                for (key_bytes, value_bytes) in user_cursor.iter_from(b"user:".as_ref()) {
                    if !key_bytes.starts_with(b"user:") {
                        break;
                    }

                    match serde_cbor::from_slice::<RbacUser>(value_bytes) {
                        Ok(mut user) => {
                            let initial_len = user.collection_roles.len();
                            user.collection_roles
                                .retain(|(col_id, _)| *col_id != collection_id);
                            if user.collection_roles.len() < initial_len {
                                users_to_update.push((key_bytes.to_vec(), user));
                            }
                        }
                        Err(e) => {
                            log::error!(
                                "Failed deserializing user {:?} during collection cleanup scan: {}",
                                String::from_utf8_lossy(key_bytes),
                                e
                            );
                        }
                    }
                }
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed open user RO cursor for cleanup: {}",
                    e
                )));
            }
        }

        log::info!(
            "Applying updates to {} user records for collection ID {} cleanup.",
            users_to_update.len(),
            collection_id
        );
        for (user_key_bytes, updated_user) in users_to_update {
            match serde_cbor::to_vec(&updated_user) {
                Ok(updated_bytes) => {
                    if let Err(e) = txn.put(
                        self.user_db,
                        &user_key_bytes,
                        &updated_bytes,
                        WriteFlags::empty(),
                    ) {
                        log::error!(
                            "Failed writing updated user {:?} during collection cleanup: {}",
                            String::from_utf8_lossy(&user_key_bytes),
                            e
                        );
                        txn.abort();
                        return Err(WaCustomError::DatabaseError(format!(
                            "Failed writing updated user: {}",
                            e
                        )));
                    }
                }
                Err(e) => {
                    log::error!(
                        "Failed serializing updated user {:?} during collection cleanup: {}",
                        String::from_utf8_lossy(&user_key_bytes),
                        e
                    );
                    txn.abort();
                    return Err(WaCustomError::SerializationError(format!(
                        "Failed serializing updated user: {}",
                        e
                    )));
                }
            }
        }
        log::info!(
            "Finished user update application for collection ID {} cleanup.",
            collection_id
        );

        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!(
                "Failed to commit collection delete and cleanup: {}",
                e
            ))
        })?;

        Ok(())
    }

    pub fn remove_user_roles_for_collection(
        &self,
        username: &str,
        collection_name: &str,
    ) -> Result<(), WaCustomError> {
        let mut txn = self.env.begin_rw_txn().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed to start RW txn for remove_roles: {}", e))
        })?;

        let collection_name_bytes = collection_name.as_bytes().to_vec();
        let collection_id_bytes = match txn.get(self.collection_db, &collection_name_bytes) {
            Ok(bytes) => bytes.to_vec(),
            Err(lmdb::Error::NotFound) => {
                return Err(WaCustomError::NotFound(format!(
                    "Collection '{}' not found",
                    collection_name
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get collection mapping: {}",
                    e
                )));
            }
        };
        let collection_id_to_remove = match collection_id_bytes
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
        {
            Ok(id) => id,
            Err(_) => {
                return Err(WaCustomError::DeserializationError(
                    "Invalid collection ID format".to_string(),
                ))
            }
        };

        let username_bytes = username.as_bytes().to_vec();
        let user_id_bytes = match txn.get(self.user_db, &username_bytes) {
            Ok(bytes) => bytes.to_vec(),
            Err(lmdb::Error::NotFound) => {
                return Err(WaCustomError::NotFound(format!(
                    "User '{}' not found",
                    username
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed to get user mapping: {}",
                    e
                )));
            }
        };
        let user_id = match user_id_bytes.as_slice().try_into().map(u32::from_le_bytes) {
            Ok(id) => id,
            Err(_) => {
                return Err(WaCustomError::DeserializationError(
                    "Invalid user ID format".to_string(),
                ))
            }
        };

        let user_key = entity_key("user", user_id);
        let mut user: RbacUser = match txn.get(self.user_db, &user_key) {
            Ok(bytes) => match serde_cbor::from_slice(bytes) {
                Ok(u) => u,
                Err(e) => {
                    return Err(WaCustomError::DeserializationError(format!(
                        "Failed deserialize user data: {}",
                        e
                    )))
                }
            },
            Err(lmdb::Error::NotFound) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Inconsistent state: User record missing for '{}'.",
                    username
                )));
            }
            Err(e) => {
                return Err(WaCustomError::DatabaseError(format!(
                    "Failed get user record: {}",
                    e
                )));
            }
        };

        let initial_len = user.collection_roles.len();
        user.collection_roles
            .retain(|(col_id, _)| *col_id != collection_id_to_remove);
        let roles_were_removed = user.collection_roles.len() < initial_len;

        if roles_were_removed {
            log::info!(
                "Removing roles for user '{}' on collection '{}' (ID {}).",
                username,
                collection_name,
                collection_id_to_remove
            );
            let updated_user_bytes = match serde_cbor::to_vec(&user) {
                Ok(bytes) => bytes,
                Err(e) => {
                    return Err(WaCustomError::SerializationError(format!(
                        "Failed serialize updated user: {}",
                        e
                    )));
                }
            };

            txn.put(
                self.user_db,
                &user_key,
                &updated_user_bytes,
                WriteFlags::empty(),
            )
            .map_err(|e| {
                WaCustomError::DatabaseError(format!("Failed put updated user data: {}", e))
            })?;
        } else {
            log::info!(
                "No roles found for user '{}' on collection '{}' (ID {}). No update needed.",
                username,
                collection_name,
                collection_id_to_remove
            );
            txn.abort();
            return Ok(());
        }
        txn.commit().map_err(|e| {
            WaCustomError::DatabaseError(format!("Failed commit role removal: {}", e))
        })?;

        Ok(())
    }
}
