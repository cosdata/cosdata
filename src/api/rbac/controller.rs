use super::dtos::*;
use crate::app_context::AppContext;
use crate::models::common::WaCustomError;
use crate::models::crypto::DoubleSHA256Hash;
use crate::rbac::store::RbacStore;
use crate::rbac::{RbacCollection, RbacUser, Role};
use actix_web::{web, HttpResponse, Responder};
use std::fmt::Write;

async fn list_all_users(store: &RbacStore) -> Result<Vec<RbacUser>, WaCustomError> {
    store.get_all_users()
}

fn convert_user_to_response(
    store: &RbacStore,
    user: RbacUser,
) -> Result<UserResponse, WaCustomError> {
    let mut collection_roles_info = Vec::new();

    for (collection_id, role_id) in user.collection_roles {
        let collection_name = match store.get_collection_by_id(collection_id)? {
            Some(c) => c.collection_name,
            None => {
                log::warn!(
                    "Collection ID {} not found while converting user {} response",
                    collection_id,
                    user.username
                );
                continue;
            }
        };

        let role_name = match store.get_role_by_id(role_id)? {
            Some(r) => r.role_name,
            None => {
                log::warn!(
                    "Role ID {} not found while converting user {} response",
                    role_id,
                    user.username
                );
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

// --- User Handlers ---
pub(crate) async fn list_users(app_ctx: web::Data<AppContext>) -> impl Responder {
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

    let user_responses_result: Result<Vec<_>, _> = users
        .into_iter()
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

pub(crate) async fn create_user(
    app_ctx: web::Data<AppContext>,
    user_data: web::Json<CreateUserRequest>,
) -> impl Responder {
    let rbac_store = &app_ctx.rbac_store;
    let auth_user_map = &app_ctx.ain_env.users_map;

    if auth_user_map.get_user(&user_data.username).is_some() {
        return HttpResponse::Conflict().json(serde_json::json!({
            "error": "conflict",
            "message": format!("User '{}' already exists in authentication system", user_data.username)
        }));
    }

    match rbac_store.get_user_by_username(&user_data.username) {
        Ok(Some(_)) => {
            log::warn!("User '{}' exists in RBAC store but was not found in auth store. This indicates a potential inconsistency.", user_data.username);
        }
        Ok(None) => {}
        Err(e) => {
            log::error!(
                "Error checking RBAC user existence for '{}': {}",
                user_data.username,
                e
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": "Failed to check RBAC user existence"
            }));
        }
    }

    let password_hash_struct = DoubleSHA256Hash::new(user_data.password.as_bytes());
    let mut password_hash_string = String::with_capacity(password_hash_struct.0.len() * 2);
    for byte in password_hash_struct.0.iter() {
        write!(&mut password_hash_string, "{:02x}", byte).unwrap();
    }

    let rbac_user_to_create = RbacUser {
        user_id: 0,
        username: user_data.username.clone(),
        password_hash: password_hash_string,
        collection_roles: Vec::new(),
    };

    let created_rbac_user = match rbac_store.create_user(rbac_user_to_create) {
        Ok(user) => user,
        Err(e) => {
            log::error!(
                "Failed to create RBAC user entry for '{}': {}",
                user_data.username,
                e
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to create RBAC user: {}", e)
            }));
        }
    };

    if let Err(e) = auth_user_map.add_user(user_data.username.clone(), password_hash_struct) {
        log::error!("Failed to create Authentication user entry for '{}': {}. RBAC user was created. Manual cleanup may be needed.", user_data.username, e);
        return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": "internal_server_error",
            "message": format!("Failed to create authentication user after RBAC user creation: {}", e)
        }));
    }

    match convert_user_to_response(rbac_store, created_rbac_user) {
        Ok(user_response) => HttpResponse::Created().json(user_response),
        Err(e) => {
            log::error!(
                "Failed formatting created user response for '{}': {}",
                user_data.username,
                e
            );
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("User created, but failed to format response: {}", e)
            }))
        }
    }
}

pub(crate) async fn get_user(
    app_ctx: web::Data<AppContext>,
    username: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let username_str = username.into_inner();

    match store.get_user_by_username(&username_str) {
        Ok(Some(user)) => match convert_user_to_response(store, user) {
            Ok(user_response) => HttpResponse::Ok().json(user_response),
            Err(e) => {
                log::error!(
                    "Failed formatting user response for '{}': {}",
                    username_str,
                    e
                );
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "internal_server_error",
                    "message": format!("Failed to format user response: {}", e)
                }))
            }
        },
        Ok(None) => HttpResponse::NotFound().json(serde_json::json!({
            "error": "not_found",
            "message": format!("User '{}' not found", username_str)
        })),
        Err(e) => {
            log::error!("Failed getting user '{}': {}", username_str, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get user: {}", e)
            }))
        }
    }
}

pub(crate) async fn update_user(
    app_ctx: web::Data<AppContext>,
    username_path: web::Path<String>,
    user_data: web::Json<UpdateUserRequest>,
) -> impl Responder {
    let rbac_store = &app_ctx.rbac_store;
    let auth_user_map = &app_ctx.ain_env.users_map;
    let original_username = username_path.into_inner();

    let mut current_rbac_user = match rbac_store.get_user_by_username(&original_username) {
        Ok(Some(user)) => user,
        Ok(None) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("User '{}' not found", original_username)
            }));
        }
        Err(e) => {
            log::error!(
                "Failed getting RBAC user '{}' for update: {}",
                original_username,
                e
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get user for update: {}", e)
            }));
        }
    };

    let original_auth_user = match auth_user_map.get_user(&original_username) {
        Some(user) => user,
        None => {
            log::error!(
                "Inconsistency: RBAC user '{}' found, but no corresponding auth user.",
                original_username
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": "User data inconsistency."
            }));
        }
    };

    let mut rbac_user_changed = false;
    let mut auth_password_hash_to_use = original_auth_user.password_hash.clone();

    if let Some(new_password_str) = &user_data.password {
        if new_password_str.is_empty() {
            return HttpResponse::BadRequest().json(serde_json::json!({
                "error": "bad_request",
                "message": "Password cannot be empty if provided"
            }));
        }
        let new_password_hash_struct = DoubleSHA256Hash::new(new_password_str.as_bytes());
        let mut new_password_hash_hex_string =
            String::with_capacity(new_password_hash_struct.0.len() * 2);
        for byte in new_password_hash_struct.0.iter() {
            write!(&mut new_password_hash_hex_string, "{:02x}", byte).unwrap();
        }
        current_rbac_user.password_hash = new_password_hash_hex_string;
        auth_password_hash_to_use = new_password_hash_struct;
        rbac_user_changed = true;
    }

    let mut username_actually_changed = false;
    let mut final_username = original_username.clone();

    if let Some(new_username_str) = &user_data.new_username {
        if new_username_str.is_empty() {
            return HttpResponse::BadRequest().json(serde_json::json!({
                "error": "bad_request",
                "message": "New username cannot be empty if provided"
            }));
        }
        if new_username_str != &original_username {
            let auth_conflict = auth_user_map.get_user(new_username_str).is_some();

            let rbac_conflict = rbac_store.get_user_by_username(new_username_str)
                .map_err(|e| {
                    log::warn!(
                        "Error during RBAC username conflict check for '{}': {}. Assuming no conflict.",
                        new_username_str, e
                    );
                    e
                })
                .ok()       // Result<Option<User>, Error> -> Option<Option<User>>
                .flatten()  // Option<Option<User>> -> Option<User>
                .is_some(); // Option<User> -> bool

            if auth_conflict || rbac_conflict {
                return HttpResponse::Conflict().json(serde_json::json!({
                    "error": "conflict",
                    "message": format!("New username '{}' is already taken", new_username_str)
                }));
            }
            current_rbac_user.username = new_username_str.clone();
            final_username = new_username_str.clone();
            username_actually_changed = true;
            rbac_user_changed = true;
        }
    }

    if rbac_user_changed {
        if let Err(e) = rbac_store.update_user(current_rbac_user.clone()) {
            log::error!(
                "Failed updating RBAC user record for '{}': {}",
                original_username,
                e
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to update RBAC user: {}", e)
            }));
        }

        if username_actually_changed {
            if let Err(e) = auth_user_map.delete_user(&original_username) {
                log::error!("Failed to delete old auth user entry for '{}' during rename: {}. RBAC user updated. Manual cleanup may be needed.", original_username, e);
            }
            if let Err(e) =
                auth_user_map.add_user(final_username.clone(), auth_password_hash_to_use.clone())
            {
                log::error!("Failed to add new auth user entry for '{}' during rename: {}. RBAC user updated. Manual cleanup may be needed.", final_username, e);
                return HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "internal_server_error",
                    "message": "Failed to update authentication user during username change."
                }));
            }
        } else if user_data.password.is_some() {
            if let Err(e) =
                auth_user_map.add_user(final_username.clone(), auth_password_hash_to_use.clone())
            {
                log::error!(
                    "Failed to update password in authentication store for user '{}': {}",
                    final_username,
                    e
                );
                return HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "internal_server_error",
                    "message": "Failed to update authentication user password."
                }));
            }
        }
    }

    let user_for_response = if rbac_user_changed {
        current_rbac_user
    } else {
        rbac_store
            .get_user_by_username(&final_username)
            .unwrap_or(None)
            .unwrap_or(current_rbac_user)
    };

    match convert_user_to_response(rbac_store, user_for_response) {
        Ok(user_response) => HttpResponse::Ok().json(user_response),
        Err(e) => {
            log::error!(
                "Failed formatting updated user response for '{}': {}",
                final_username,
                e
            );
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("User update processed, but failed to format response: {}", e)
            }))
        }
    }
}

pub(crate) async fn delete_user(
    app_ctx: web::Data<AppContext>,
    username: web::Path<String>,
) -> impl Responder {
    let rbac_store = &app_ctx.rbac_store;
    let auth_user_map = &app_ctx.ain_env.users_map;
    let username_str = username.into_inner();

    match rbac_store.delete_user_by_username(&username_str) {
        Ok(()) => {
            log::info!(
                "Successfully deleted user '{}' from RBAC store.",
                username_str
            );
        }
        Err(WaCustomError::NotFound(msg)) => {
            log::info!(
                "User '{}' not found in RBAC store for deletion.",
                username_str
            );
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": msg
            }));
        }
        Err(e) => {
            log::error!(
                "Failed deleting user '{}' from RBAC store: {}",
                username_str,
                e
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to delete user from RBAC store: {}", e)
            }));
        }
    }

    if let Err(e) = auth_user_map.delete_user(&username_str) {
        log::error!("Failed to delete user '{}' from authentication store. RBAC user was deleted. Manual cleanup may be needed: {}", username_str, e);
        return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": "partial_failure",
            "message": format!("User '{}' deleted from RBAC, but failed to delete from authentication store: {}. Please check logs.", username_str, e)
        }));
    }

    log::info!(
        "Successfully deleted user '{}' from authentication store.",
        username_str
    );
    HttpResponse::Ok().json(serde_json::json!({
        "message": format!("User '{}' successfully deleted from all systems", username_str)
    }))
}

pub(crate) async fn list_roles(app_ctx: web::Data<AppContext>) -> impl Responder {
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

pub(crate) async fn create_role(
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
        }
        Ok(None) => {}
        Err(e) => {
            log::error!(
                "Failed checking role existence for '{}': {}",
                role_data.role_name,
                e
            );
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
        }
        Err(e) => {
            log::error!("Failed creating role '{}': {}", role_data.role_name, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to create role: {}", e)
            }))
        }
    }
}

pub(crate) async fn get_role(
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
        Ok(None) => HttpResponse::NotFound().json(serde_json::json!({
            "error": "not_found",
            "message": format!("Role '{}' not found", name_to_find)
        })),
        Err(e) => {
            log::error!("Failed getting role '{}': {}", name_to_find, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to retrieve role: {}", e)
            }))
        }
    }
}

pub(crate) async fn update_role(
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
            Ok(None) => {}
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
            log::error!(
                "Role ID {} not found during update_role call, inconsistent state?",
                existing_role.role_id
            );
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

pub(crate) async fn delete_role(
    app_ctx: web::Data<AppContext>,
    role_name_path: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let name_to_delete = role_name_path.into_inner();

    match store.delete_role_by_name(&name_to_delete) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({
            "message": format!("Role '{}' successfully deleted", name_to_delete)
        })),
        Err(WaCustomError::NotFound(msg)) => HttpResponse::NotFound().json(serde_json::json!({
            "error": "not_found",
            "message": msg
        })),
        Err(e) => {
            log::error!("Failed to delete role '{}': {}", name_to_delete, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to delete role: {}", e)
            }))
        }
    }
}

// --- RBAC Collection and Assignment Handlers ---
pub(crate) async fn list_rbac_collections(app_ctx: web::Data<AppContext>) -> impl Responder {
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

pub(crate) async fn create_rbac_collection(
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
        }
        Ok(None) => {}
        Err(e) => {
            log::error!(
                "Failed checking RBAC collection existence for '{}': {}",
                name_to_create,
                e
            );
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
            log::error!(
                "Failed creating RBAC collection '{}': {}",
                name_to_create,
                e
            );
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to create RBAC collection: {}", e)
            }))
        }
    }
}

pub(crate) async fn delete_rbac_collection(
    app_ctx: web::Data<AppContext>,
    path: web::Path<String>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let name_or_id_to_delete = path.into_inner();

    match store.delete_collection_by_name(&name_or_id_to_delete) {
        Ok(()) => {
            HttpResponse::Ok().json(serde_json::json!({
                "message": format!("RBAC Collection entry '{}' successfully deleted", name_or_id_to_delete)
            }))
        }
        Err(WaCustomError::NotFound(msg)) => {
             HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": msg
            }))
        }
        Err(e) => {
            log::error!("Failed to delete RBAC collection entry '{}': {}", name_or_id_to_delete, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to delete RBAC collection entry: {}", e)
            }))
        }
    }
}

pub(crate) async fn assign_role(
    app_ctx: web::Data<AppContext>,
    path: web::Path<(String, String, String)>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let (collection_id_or_name, username, role_name) = path.into_inner();

    let rbac_collection = match store.get_collection_by_id_or_name(&collection_id_or_name) {
        Ok(Some(collection)) => collection,
        Ok(None) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("RBAC Collection '{}' not found", collection_id_or_name)
            }));
        }
        Err(e) => {
            log::error!(
                "Failed getting RBAC collection '{}' for role assignment: {}",
                collection_id_or_name,
                e
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get RBAC collection: {}", e)
            }));
        }
    };

    match store.assign_role_to_user(&username, &rbac_collection.collection_name, &role_name) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({
            "message": format!("Successfully assigned role '{}' to user '{}' for collection '{}'",
                               role_name, username, rbac_collection.collection_name)
        })),
        Err(WaCustomError::NotFound(msg)) => HttpResponse::NotFound().json(serde_json::json!({
            "error": "not_found",
            "message": msg
        })),
        Err(e) => {
            log::error!(
                "Failed assigning role '{}' to user '{}' for collection '{}': {}",
                role_name,
                username,
                rbac_collection.collection_name,
                e
            );
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to assign role: {}", e)
            }))
        }
    }
}

pub(crate) async fn remove_user_roles(
    app_ctx: web::Data<AppContext>,
    path: web::Path<(String, String)>,
) -> impl Responder {
    let store = &app_ctx.rbac_store;
    let (collection_id_or_name, username) = path.into_inner();

    let rbac_collection = match store.get_collection_by_id_or_name(&collection_id_or_name) {
        Ok(Some(collection)) => collection,
        Ok(None) => {
            return HttpResponse::NotFound().json(serde_json::json!({
                "error": "not_found",
                "message": format!("RBAC Collection '{}' not found", collection_id_or_name)
            }));
        }
        Err(e) => {
            log::error!(
                "Failed getting RBAC collection '{}' for role removal: {}",
                collection_id_or_name,
                e
            );
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to get RBAC collection: {}", e)
            }));
        }
    };

    match store.remove_user_roles_for_collection(&username, &rbac_collection.collection_name) {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({
            "message": format!("All roles successfully removed for user '{}' from collection '{}'",
                               username, rbac_collection.collection_name)
        })),
        Err(WaCustomError::NotFound(msg)) => HttpResponse::NotFound().json(serde_json::json!({
            "error": "not_found",
            "message": msg
        })),
        Err(e) => {
            log::error!(
                "Failed removing roles for user '{}' from collection '{}': {}",
                username,
                rbac_collection.collection_name,
                e
            );
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": format!("Failed to remove roles: {}", e)
            }))
        }
    }
}
