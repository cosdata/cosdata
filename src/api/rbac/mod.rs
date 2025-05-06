use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

use crate::app_context::AppContext;
use crate::models::common::WaCustomError;
use crate::rbac::{Permission, RbacCollection, RbacUser, Role};
use crate::rbac::guards::require_manage_permissions;
use crate::models::crypto::DoubleSHA256Hash;
use crate::rbac::store::RbacStore;

#[derive(Serialize, Deserialize)]
struct CreateUserRequest {
    username: String,
    password: String,
}

#[derive(Serialize, Deserialize)]
struct UpdateUserRequest {
    new_username: Option<String>,
    password: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CollectionRoleInfo {
    collection_id: u32,
    collection_name: String,
    role_id: u32,
    role_name: String,
}

#[derive(Serialize, Deserialize)]
struct UserResponse {
    user_id: u32,
    username: String,
    collection_roles: Vec<CollectionRoleInfo>,
}

#[derive(Serialize, Deserialize)]
struct CreateRoleRequest {
    role_name: String,
    permissions: Vec<Permission>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RoleResponse {
    role_id: u32,
    role_name: String,
    permissions: Vec<Permission>,
}

#[derive(Serialize, Deserialize)]
struct CreateCollectionRequest {
    collection_name: String,
}

#[derive(Serialize, Deserialize)]
struct CollectionResponse {
    collection_id: u32,
    collection_name: String,
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        // Apply guard to the whole scope
        web::scope("/rbac")
            .wrap(require_manage_permissions())
            .service(
                web::scope("/users")
                    .route("", web::get().to(list_users))
                    .route("", web::post().to(create_user))
                    .route("/{username}", web::get().to(get_user))
                    .route("/{username}", web::put().to(update_user))
                    .route("/{username}", web::delete().to(delete_user))
            )
            .service(
                web::scope("/roles")
                    .route("", web::get().to(list_roles))
                    .route("", web::post().to(create_role))
                    .route("/{role_name}", web::get().to(get_role))
                    .route("/{role_name}", web::put().to(update_role))
                    .route("/{role_name}", web::delete().to(delete_role))
            )
            .service(
                 web::scope("/collections")
                    .route("", web::get().to(list_rbac_collections))
                    .route("", web::post().to(create_rbac_collection))
                    .route("/{collection_id_or_name}", web::delete().to(delete_rbac_collection))
                    .route("/{collection_id_or_name}/users/{username}/roles/{role_name}", web::put().to(assign_role))
                    .route("/{collection_id_or_name}/users/{username}/roles", web::delete().to(remove_user_roles))
            )
    );
}

async fn list_all_users(store: &RbacStore) -> Result<Vec<RbacUser>, WaCustomError> {
    store.get_all_users()
}

// Helper function to convert a user to a response DTO, fetching associated collection and role names
fn convert_user_to_response(store: &RbacStore, user: RbacUser) -> Result<UserResponse, WaCustomError> {
    let mut collection_roles_info = Vec::new();

    for (collection_id, role_id) in user.collection_roles {
        let collection_name = match store.get_collection_by_id(collection_id)? {
            Some(c) => c.collection_name,
            None => {
                log::warn!("Collection ID {} not found while converting user {} response", collection_id, user.username);
                continue;
            }
        };

        let role_name = match store.get_role_by_id(role_id)? {
            Some(r) => r.role_name,
            None => {
                 log::warn!("Role ID {} not found while converting user {} response", role_id, user.username);
                 continue;
            }
        };

        collection_roles_info.push(CollectionRoleInfo {
            collection_id,
            collection_name,
            role_id,
            role_name,
        });
    }

    Ok(UserResponse {
        user_id: user.user_id,
        username: user.username,
        collection_roles: collection_roles_info,
    })
}


async fn list_users(app_ctx: web::Data<AppContext>) -> impl Responder {
    let store = &app_ctx.rbac_store;

    let users = match list_all_users(store).await {
        Ok(users) => users,
        Err(e) => {
            log::error!("Failed to list users: {}", e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to list users: {}", e)
            }));
        }
    };

    let user_responses_result: Result<Vec<_>, _> = users.into_iter()
        .map(|user| convert_user_to_response(store, user))
        .collect();

    match user_responses_result {
         Ok(user_responses) => HttpResponse::Ok().json(user_responses),
         Err(e) => {
              log::error!("Error converting users to response: {}", e);
              HttpResponse::InternalServerError().json(serde_json::json!({
                  "error": "internal_server_error",
                   "message": format!("Failed to format user list response: {}", e)
              }))
         }
    }
}

async fn create_user(
    app_ctx: web::Data<AppContext>,
    user_data: web::Json<CreateUserRequest>,
) -> impl Responder {
    let rbac_store = &app_ctx.rbac_store;
    let auth_user_map = &app_ctx.ain_env.users_map;

    // Check existence in authentication store first
    match auth_user_map.get_user(&user_data.username) {
        Some(_) => {
             return HttpResponse::Conflict().json(serde_json::json!({
                "error": "conflict",
                "message": format!("User '{}' already exists for authentication", user_data.username)
            }));
        }
        None => {}
    }
    // Optionally check RBAC store for consistency
    match rbac_store.get_user_by_username(&user_data.username) {
         Ok(Some(_)) => {
              log::warn!("User '{}' exists in RBAC store but not auth store. Proceeding but state might be inconsistent.", user_data.username);
         }
         Ok(None) => {}
         Err(e) => { log::error!("Error checking RBAC user existence: {}", e); }
    }

    let password_hash_struct = DoubleSHA256Hash::new(user_data.password.as_bytes());
    let password_hash_string: String = password_hash_struct.0.iter().map(|b| format!("{:02x}", b)).collect();

    let rbac_user_to_create = RbacUser {
        user_id: 0,
        username: user_data.username.clone(),
        password_hash: password_hash_string,
        collection_roles: Vec::new(),
    };
    let created_rbac_user = match rbac_store.create_user(rbac_user_to_create) {
         Ok(user) => user,
         Err(e) => {
             log::error!("Failed to create RBAC user entry for '{}': {}", user_data.username, e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to create RBAC user: {}", e)
            }));
         }
    };

    // Create the Authentication user entry (using the DoubleSHA256Hash struct directly)
    if let Err(e) = auth_user_map.add_user(user_data.username.clone(), password_hash_struct) {
         log::error!("Failed to create Authentication user entry for '{}': {}", user_data.username, e);
         return HttpResponse::InternalServerError().json(serde_json::json!({
             "error": "internal_server_error",
             "message": format!("Failed to create authentication user: {}", e)
         }));
    }

    match convert_user_to_response(rbac_store, created_rbac_user) {
        Ok(user_response) => HttpResponse::Created().json(user_response),
        Err(e) => {
             log::error!("Failed formatting created user response for '{}': {}", user_data.username, e);
              HttpResponse::InternalServerError().json(serde_json::json!({
                 "error": "internal_server_error",
                 "message": format!("User created, but failed to format response: {}", e)
             }))
        }
    }
}

async fn get_user(
    app_ctx: web::Data<AppContext>,
    username: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let username_str = username.into_inner();

    match store.get_user_by_username(&username_str) {
        Ok(Some(user)) => {
            match convert_user_to_response(store, user) {
                Ok(user_response) => {
                    HttpResponse::Ok().json(user_response)
                },
                Err(e) => {
                    log::error!("Failed formatting user response for '{}': {}", username_str, e);
                     HttpResponse::InternalServerError().json(serde_json::json!({
                        "error": "internal_server_error",
                        "message": format!("Failed to format user response: {}", e)
                    }))
                }
            }
        },
        Ok(None) => {
            HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("User '{}' not found", username_str)
            }))
        },
        Err(e) => {
             log::error!("Failed getting user '{}': {}", username_str, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get user: {}", e)
            }))
        }
    }
}

async fn update_user(
    app_ctx: web::Data<AppContext>,
    username: web::Path<String>,
    user_data: web::Json<UpdateUserRequest>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let username_str = username.into_inner();

    let mut user = match store.get_user_by_username(&username_str) {
        Ok(Some(user)) => user,
        Ok(None) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("User '{}' not found", username_str)
            }));
        },
        Err(e) => {
             log::error!("Failed getting user '{}' for update: {}", username_str, e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get user: {}", e)
            }));
        }
    };

    if let Some(new_password) = &user_data.password {
        if !new_password.is_empty() {
             let hash = DoubleSHA256Hash::new(new_password.as_bytes());
             user.password_hash = hash.0.iter().map(|b| format!("{:02x}", b)).collect();
        } else {
             return HttpResponse::BadRequest().json(serde_json::json!({
                 "error": "bad_request",
                 "message": "Password cannot be empty"
             }));
        }
    }

    let mut username_changed = false;
    if let Some(new_username) = &user_data.new_username {
         if !new_username.is_empty() && new_username != &user.username {
             // Check if the new username is already taken
             match store.get_user_by_username(new_username) {
                 Ok(Some(_)) => {
                     return HttpResponse::Conflict().json(serde_json::json!({
                         "error": "conflict",
                         "message": format!("Username '{}' is already taken", new_username)
                     }));
                 },
                 Ok(None) => {
                     user.username = new_username.clone();
                     username_changed = true;
                 },
                 Err(e) => {
                      log::error!("Failed checking new username availability for '{}': {}", new_username, e);
                     return HttpResponse::InternalServerError().json(serde_json::json!({
                         "error": "internal_server_error",
                         "message": format!("Failed to check username availability: {}", e)
                     }));
                 }
             }
         } else if new_username.is_empty() {
              return HttpResponse::BadRequest().json(serde_json::json!({
                  "error": "bad_request",
                  "message": "Username cannot be empty"
              }));
         }
    }

    // Only call update if something actually changed (password or username)
    if user_data.password.is_some() || username_changed {
        match store.update_user(user.clone()) {
            Ok(()) => { }
            Err(e) => {
                 log::error!("Failed updating user '{}': {}", username_str, e);
                return HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "internal_server_error",
                    "message": format!("Failed to update user: {}", e)
                }));
            }
        }
    }

    match convert_user_to_response(store, user) {
        Ok(user_response) => {
            HttpResponse::Ok().json(user_response)
        },
        Err(e) => {
            log::error!("Failed formatting updated user response for '{}': {}", username_str, e);
             HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("User updated (or unchanged), but failed to format response: {}", e)
            }))
        }
    }
}

async fn delete_user(
    app_ctx: web::Data<AppContext>,
    username: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let username_str = username.into_inner();

    match store.delete_user_by_username(&username_str) {
        Ok(()) => {
            HttpResponse::Ok().json(serde_json::json!({
                "message": format!("User '{}' successfully deleted", username_str)
            }))
        },
        Err(WaCustomError::NotFound(msg)) => {
             HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": msg
            }))
        }
        Err(e) => {
            log::error!("Failed deleting user '{}': {}", username_str, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to delete user: {}", e)
            }))
        }
    }
}

async fn list_roles(app_ctx: web::Data<AppContext>) -> impl Responder {
    let store = &app_ctx.rbac_store;

    match store.get_all_roles() {
        Ok(roles) => {
            let response_list: Vec<RoleResponse> = roles
                .into_iter()
                .map(|role| RoleResponse {
                    role_id: role.role_id,
                    role_name: role.role_name,
                    permissions: role.permissions,
                })
                .collect();
            HttpResponse::Ok().json(response_list)
        }
        Err(e) => {
            log::error!("Failed to list roles: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to retrieve roles: {}", e)
            }))
        }
    }
}

async fn create_role(
    app_ctx: web::Data<AppContext>,
    role_data: web::Json<CreateRoleRequest>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;

    if role_data.role_name.is_empty() {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "bad_request",
            "message": "Role name cannot be empty"
        }));
    }

    match store.get_role_by_name(&role_data.role_name) {
        Ok(Some(_)) => {
            return HttpResponse::Conflict().json(serde_json::json!({
                "error": "conflict",
                "message": format!("Role '{}' already exists", role_data.role_name)
            }));
        },
        Ok(None) => { },
        Err(e) => {
             log::error!("Failed checking role existence for '{}': {}", role_data.role_name, e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to check role existence: {}", e)
            }));
        }
    }

    let role_to_create = Role {
        role_id: 0,
        role_name: role_data.role_name.clone(),
        permissions: role_data.permissions.clone(),
    };

    match store.create_role(role_to_create) {
        Ok(created_role) => {
            let role_response = RoleResponse {
                role_id: created_role.role_id,
                role_name: created_role.role_name,
                permissions: created_role.permissions,
            };
            HttpResponse::Created().json(role_response)
        },
        Err(e) => {
            log::error!("Failed creating role '{}': {}", role_data.role_name, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to create role: {}", e)
            }))
        }
    }
}

async fn get_role(
    app_ctx: web::Data<AppContext>,
    role_name: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let name_to_find = role_name.into_inner();

    match store.get_role_by_name(&name_to_find) {
        Ok(Some(role)) => {
            let response = RoleResponse {
                role_id: role.role_id,
                role_name: role.role_name,
                permissions: role.permissions,
            };
            HttpResponse::Ok().json(response)
        }
        Ok(None) => {
            HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("Role '{}' not found", name_to_find)
            }))
        }
        Err(e) => {
            log::error!("Failed getting role '{}': {}", name_to_find, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to retrieve role: {}", e)
            }))
        }
    }
}

async fn update_role(
    app_ctx: web::Data<AppContext>,
    role_name_path: web::Path<String>,
    role_data: web::Json<CreateRoleRequest>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let name_to_update = role_name_path.into_inner();

    let existing_role = match store.get_role_by_name(&name_to_update) {
        Ok(Some(role)) => role,
        Ok(None) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("Role '{}' not found", name_to_update)
            }));
        }
        Err(e) => {
            log::error!("Failed to get role '{}' for update: {}", name_to_update, e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to retrieve role: {}", e)
            }));
        }
    };

    if existing_role.role_name != role_data.role_name {
         if role_data.role_name.is_empty() {
             return HttpResponse::BadRequest().json(serde_json::json!({
                "error": "bad_request",
                "message": "Role name cannot be empty"
             }));
         }
        match store.get_role_by_name(&role_data.role_name) {
            Ok(Some(_)) => {
                return HttpResponse::Conflict().json(serde_json::json!({
                    "error": "conflict",
                    "message": format!("Role name '{}' is already taken", role_data.role_name)
                }));
            }
            Ok(None) => { }
            Err(e) => {
                log::error!("Failed checking new role name availability: {}", e);
                 return HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "internal_server_error",
                    "message": format!("Failed checking role name: {}", e)
                }));
            }
        }
    }

    let updated_role = Role {
        role_id: existing_role.role_id,
        role_name: role_data.role_name.clone(),
        permissions: role_data.permissions.clone(),
    };

    match store.update_role(updated_role.clone()) {
        Ok(()) => {
            let response = RoleResponse {
                role_id: updated_role.role_id,
                role_name: updated_role.role_name,
                permissions: updated_role.permissions,
            };
            HttpResponse::Ok().json(response)
        }
        Err(WaCustomError::NotFound(msg)) => {
             log::error!("Role ID {} not found during update_role call, inconsistent state?", existing_role.role_id);
             HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Inconsistent state: {}", msg)
            }))
        }
        Err(e) => {
            log::error!("Failed to update role '{}': {}", name_to_update, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to update role: {}", e)
            }))
        }
    }
}

async fn delete_role(
    app_ctx: web::Data<AppContext>,
    role_name_path: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let name_to_delete = role_name_path.into_inner();

    match store.delete_role_by_name(&name_to_delete) {
        Ok(()) => {
            HttpResponse::Ok().json(serde_json::json!({
                "message": format!("Role '{}' successfully deleted", name_to_delete)
            }))
        }
        Err(WaCustomError::NotFound(msg)) => {
             HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": msg
            }))
        }
        Err(e) => {
            log::error!("Failed to delete role '{}': {}", name_to_delete, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to delete role: {}", e)
            }))
        }
    }
}

async fn list_rbac_collections(app_ctx: web::Data<AppContext>) -> impl Responder {
    let store = &app_ctx.rbac_store;

    match store.get_all_collections() {
        Ok(collections) => {
            let response_list: Vec<CollectionResponse> = collections
                .into_iter()
                .map(|coll| CollectionResponse {
                    collection_id: coll.collection_id,
                    collection_name: coll.collection_name,
                })
                .collect();
            HttpResponse::Ok().json(response_list)
        }
        Err(e) => {
            log::error!("Failed to list RBAC collections: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to retrieve RBAC collections: {}", e)
            }))
        }
    }
}

async fn create_rbac_collection(
    app_ctx: web::Data<AppContext>,
    collection_data: web::Json<CreateCollectionRequest>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let name_to_create = &collection_data.collection_name;

    if name_to_create.is_empty() {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "bad_request",
            "message": "Collection name cannot be empty"
        }));
    }

    match store.get_collection_by_name(name_to_create) {
        Ok(Some(_)) => {
            return HttpResponse::Conflict().json(serde_json::json!({
                "error": "conflict",
                "message": format!("RBAC collection '{}' already exists", name_to_create)
            }));
        },
        Ok(None) => { }
        Err(e) => {
            log::error!("Failed checking RBAC collection existence for '{}': {}", name_to_create, e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to check collection existence: {}", e)
            }));
        }
    }

    let new_rbac_collection = RbacCollection {
        collection_id: 0,
        collection_name: name_to_create.clone(),
    };

    match store.create_collection(new_rbac_collection) {
        Ok(created_collection) => {
             let response = CollectionResponse {
                 collection_id: created_collection.collection_id,
                 collection_name: created_collection.collection_name,
             };
             HttpResponse::Created().json(response)
        }
        Err(e) => {
            log::error!("Failed creating RBAC collection '{}': {}", name_to_create, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to create RBAC collection: {}", e)
            }))
        }
    }
}

async fn remove_user_roles(
    app_ctx: web::Data<AppContext>,
    path: web::Path<(String, String)>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let (collection_id_or_name, username) = path.into_inner();

    let collection = match store.get_collection_by_id_or_name(&collection_id_or_name) {
        Ok(Some(collection)) => collection,
        Ok(None) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("Collection '{}' not found", collection_id_or_name)
            }));
        },
        Err(e) => {
            log::error!("Failed getting collection '{}' for role removal: {}", collection_id_or_name, e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get collection: {}", e)
            }));
        }
    };

    match store.remove_user_roles_for_collection(&username, &collection.collection_name) {
        Ok(()) => {
            HttpResponse::Ok().json(serde_json::json!({
                "message": format!("All roles successfully removed for user '{}' in collection '{}'",
                                   username, collection.collection_name)
            }))
        }
        Err(WaCustomError::NotFound(msg)) => {
             HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": msg
            }))
        }
         Err(e) => {
            log::error!("Failed removing roles for user '{}' in collection '{}': {}", username, collection.collection_name, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to remove roles: {}", e)
            }))
        }
    }
}

async fn assign_role(
    app_ctx: web::Data<AppContext>,
    path: web::Path<(String, String, String)>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let (collection_id_or_name, username, role_name) = path.into_inner();

    let collection = match store.get_collection_by_id_or_name(&collection_id_or_name) {
        Ok(Some(collection)) => collection,
        Ok(None) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("Collection '{}' not found", collection_id_or_name)
            }));
        },
        Err(e) => {
            log::error!("Failed getting collection '{}' for role assignment: {}", collection_id_or_name, e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get collection: {}", e)
            }));
        }
    };

    match store.assign_role_to_user(&username, &collection.collection_name, &role_name) {
        Ok(()) => {
            HttpResponse::Ok().json(serde_json::json!({
                "message": format!("Successfully assigned role '{}' to user '{}' for collection '{}'",
                                   role_name, username, collection.collection_name)
            }))
        },
        Err(WaCustomError::NotFound(msg)) => {
             HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": msg
            }))
        },
        Err(e) => {
             log::error!("Failed assigning role '{}' to user '{}' for collection '{}': {}", role_name, username, collection.collection_name, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to assign role: {}", e)
            }))
        }
    }
}

async fn delete_rbac_collection(
    app_ctx: web::Data<AppContext>,
    path: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let name_or_id_to_delete = path.into_inner();

    let collection_name = name_or_id_to_delete;

    match store.delete_collection_by_name(&collection_name) {
        Ok(()) => {
            HttpResponse::Ok().json(serde_json::json!({
                "message": format!("RBAC Collection entry '{}' successfully deleted", collection_name)
            }))
        }
        Err(WaCustomError::NotFound(msg)) => {
             HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": msg
            }))
        }
        Err(e) => {
            log::error!("Failed to delete RBAC collection entry '{}': {}", collection_name, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to delete RBAC collection entry: {}", e)
            }))
        }
    }
}
