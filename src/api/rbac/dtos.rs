use crate::rbac::Permission;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) struct CreateUserRequest {
    pub username: String,
    pub password: String,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct UpdateUserRequest {
    pub new_username: Option<String>,
    pub password: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct CollectionRoleInfo {
    pub collection_id: u32,
    pub collection_name: String,
    pub role_id: u32,
    pub role_name: String,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct UserResponse {
    pub user_id: u32,
    pub username: String,
    pub collection_roles: Vec<CollectionRoleInfo>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CreateRoleRequest {
    pub role_name: String,
    pub permissions: Vec<Permission>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RoleResponse {
    pub role_id: u32,
    pub role_name: String,
    pub permissions: Vec<Permission>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CreateCollectionRequest {
    pub collection_name: String,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CollectionResponse {
    pub collection_id: u32,
    pub collection_name: String,
}
