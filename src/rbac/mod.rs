use serde::{Deserialize, Serialize};

pub mod store;
pub mod middleware;
pub mod guards;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    ListCollections,
    CreateCollection,
    UpdateCollection,
    DeleteCollection,
    ListIndex,
    CreateIndex,
    DeleteIndex,
    UpsertVectors,
    DeleteVectors,
    ListVectors,
    CheckVectorExistence,
    QueryDenseVectors,
    QuerySparseVectors,
    QueryHybridVectors,
    ListVersions,
    SetCurrentVersion,
    GetCurrentVersion,
    ManagePermissions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub role_id: u32,
    pub role_name: String,
    pub permissions: Vec<Permission>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacUser {
    pub user_id: u32,
    pub username: String,
    pub password_hash: String,
    pub collection_roles: Vec<(u32, u32)>,  // (collection_id, role_id)
}

// Collection struct for RBAC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacCollection {
    pub collection_id: u32,
    pub collection_name: String,
}

// Role definitions for built-in roles
pub fn get_admin_role() -> Role {
    let all_permissions = vec![
        Permission::ListCollections,
        Permission::CreateCollection,
        Permission::UpdateCollection,
        Permission::DeleteCollection,

        Permission::ListIndex,
        Permission::CreateIndex,
        Permission::DeleteIndex,

        Permission::UpsertVectors,
        Permission::DeleteVectors,
        Permission::ListVectors,
        Permission::CheckVectorExistence,

        Permission::QueryDenseVectors,
        Permission::QuerySparseVectors,
        Permission::QueryHybridVectors,

        Permission::ListVersions,
        Permission::SetCurrentVersion,
        Permission::GetCurrentVersion,

        Permission::ManagePermissions,
    ];

    Role {
        role_id: 1,
        role_name: "admin".to_string(),
        permissions: all_permissions,
    }
}

pub fn get_reader_role() -> Role {
    Role {
        role_id: 2,
        role_name: "reader".to_string(),
        permissions: vec![
            Permission::ListCollections,
            Permission::ListIndex,
            Permission::ListVectors,
            Permission::CheckVectorExistence,

            Permission::QueryDenseVectors,
            Permission::QuerySparseVectors,
            Permission::QueryHybridVectors,

            Permission::ListVersions,
            Permission::GetCurrentVersion,
        ],
    }
}

pub fn get_writer_role() -> Role {
    Role {
        role_id: 3,
        role_name: "writer".to_string(),
        permissions: vec![
            Permission::ListCollections,
            Permission::UpdateCollection,

            Permission::ListIndex,

            Permission::UpsertVectors,
            Permission::ListVectors,
            Permission::CheckVectorExistence,

            Permission::QueryDenseVectors,
            Permission::QuerySparseVectors,
            Permission::QueryHybridVectors,

            Permission::ListVersions,
            Permission::GetCurrentVersion,
        ],
    }
}
