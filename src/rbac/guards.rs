use crate::rbac::{middleware::AuthorizationMiddleware, Permission};

// Collection permissions
pub fn require_list_collections() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::ListCollections,
    }
}

pub fn require_create_collection() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::CreateCollection,
    }
}

pub fn require_update_collection() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::UpdateCollection,
    }
}

pub fn require_delete_collection() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::DeleteCollection,
    }
}

// Index permissions
pub fn require_list_index() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::ListIndex,
    }
}

pub fn require_create_index() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::CreateIndex,
    }
}

pub fn require_delete_index() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::DeleteIndex,
    }
}

// Vector permissions
pub fn require_list_vectors() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::ListVectors,
    }
}

pub fn require_check_vector_existence() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::CheckVectorExistence,
    }
}

pub fn require_upsert_vectors() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::UpsertVectors,
    }
}

pub fn require_delete_vectors() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::DeleteVectors,
    }
}

// Query permissions
pub fn require_query_dense_vectors() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::QueryDenseVectors,
    }
}

pub fn require_query_sparse_vectors() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::QuerySparseVectors,
    }
}

pub fn require_query_hybrid_vectors() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::QueryHybridVectors,
    }
}

// Version permissions
pub fn require_list_versions() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::ListVersions,
    }
}

pub fn require_set_current_version() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::SetCurrentVersion,
    }
}

pub fn require_get_current_version() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::GetCurrentVersion,
    }
}

// Permission management
pub fn require_manage_permissions() -> AuthorizationMiddleware {
    AuthorizationMiddleware {
        required_permission: Permission::ManagePermissions,
    }
}
