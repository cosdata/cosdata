use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    error::Error,
    http::StatusCode,
    web, HttpMessage, HttpResponse,
};
use futures_util::future::{ready, Ready};
use futures_util::future::LocalBoxFuture;

use crate::app_context::AppContext;
use crate::api::auth::dtos::Claims;
use crate::rbac::Permission;

use crate::rbac::store::RbacStore;
use crate::models::common::WaCustomError;

// Custom error for authorization failures
#[derive(Debug)]
pub enum AuthorizationError {
    Unauthorized(String),
    MissingCollection,
    InternalError(String),
}

impl std::fmt::Display for AuthorizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
            Self::MissingCollection => write!(f, "Missing collection information"),
            Self::InternalError(msg) => write!(f, "Internal server error: {}", msg),
        }
    }
}

impl actix_web::error::ResponseError for AuthorizationError {
    fn error_response(&self) -> HttpResponse {
        match self {
            Self::Unauthorized(_) => HttpResponse::Forbidden().json(serde_json::json!({
                "error": "forbidden",
                "message": self.to_string()
            })),
            Self::MissingCollection => HttpResponse::BadRequest().json(serde_json::json!({
                "error": "bad_request",
                "message": self.to_string()
            })),
            Self::InternalError(_) => HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "internal_server_error",
                "message": self.to_string()
            })),
        }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::Unauthorized(_) => StatusCode::FORBIDDEN,
            Self::MissingCollection => StatusCode::BAD_REQUEST,
            Self::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

// Authorization middleware config that specifies the required permission
pub struct AuthorizationMiddleware {
    pub required_permission: Permission,
}

impl<S, B> Transform<S, ServiceRequest> for AuthorizationMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthorizationMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthorizationMiddlewareService {
            service,
            required_permission: self.required_permission,
        }))
    }
}

// Middleware service
pub struct AuthorizationMiddlewareService<S> {
    service: S,
    required_permission: Permission,
}

impl<S, B> Service<ServiceRequest> for AuthorizationMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let claims = match req.extensions().get::<Claims>() {
            Some(claims) => claims.clone(),
            None => {
                return Box::pin(async {
                    Err(AuthorizationError::Unauthorized("User not authenticated".to_string()).into())
                });
            }
        };

        let app_ctx = match req.app_data::<web::Data<AppContext>>() {
            Some(ctx) => ctx.clone(),
            None => {
                return Box::pin(async {
                    Err(AuthorizationError::InternalError("App context not found".to_string()).into())
                });
            }
        };

        let collection_name = match self.extract_collection_name(&req) {
            Ok(name) => name,
            Err(e) => {
                return Box::pin(async move { Err(e.into()) });
            }
        };

        let username = claims.username.clone();
        let required_permission = self.required_permission;
        let rbac_store = app_ctx.rbac_store.clone();
        let collection_name_clone = collection_name.clone();

        let service = &self.service;
        let fut = service.call(req);

        Box::pin(async move {
            let has_permission = if required_permission == Permission::ManagePermissions {
                log::debug!("Performing GLOBAL check for ManagePermissions for user '{}'", username);
                match check_global_permission(&rbac_store, &username, required_permission) {
                    Ok(result) => result,
                    Err(e) => { /* Handle error... */ return Err(AuthorizationError::InternalError(format!("Failed check global perm: {}", e)).into()); }
                }
            } else if required_permission == Permission::DeleteCollection {
                 log::debug!("Performing GLOBAL check for DeleteCollection for user '{}'", username);
                 match check_global_permission(&rbac_store, &username, required_permission) {
                     Ok(true) => {
                         log::debug!("GLOBAL DeleteCollection permission GRANTED for user '{}'", username);
                         true
                     }
                     Ok(false) => {
                         log::debug!("Global DeleteCollection check failed, checking specific collection...");
                         match &collection_name_clone {
                             Some(collection) => {
                                 log::debug!("Performing collection check for DeleteCollection on '{}' for user '{}'", collection, username);
                                 match rbac_store.check_permission(&username, collection, required_permission) {
                                     Ok(result) => result,
                                     Err(e) => { /* Handle error... */ return Err(AuthorizationError::InternalError(format!("Failed check specific perm: {}", e)).into()); }
                                 }
                             }
                             None => {
                                 log::debug!("No specific collection provided for DeleteCollection check.");
                                 false
                             }
                         }
                     }
                     Err(e) => { /* Handle error... */ return Err(AuthorizationError::InternalError(format!("Failed check global perm: {}", e)).into()); }
                 }
            } else {
                // All Other Permissions (Specific if possible, else Global)
                match &collection_name_clone {
                    Some(collection) => {
                        // Collection-specific check
                        log::debug!("Performing collection check for permission '{:?}' on collection '{}' for user '{}'",
                            required_permission, collection, username);
                        match rbac_store.check_permission(&username, collection, required_permission) {
                            Ok(result) => result,
                            Err(e) => { /* Handle error... */ return Err(AuthorizationError::InternalError(format!("Failed check specific perm: {}", e)).into()); }
                        }
                    },
                    None => {
                        // Global check for other non-ManagePermissions actions (like CreateCollection)
                         log::debug!("Performing GLOBAL check for permission '{:?}' for user '{}'",
                            required_permission, username);
                        match check_global_permission(&rbac_store, &username, required_permission) {
                            Ok(result) => result,
                            Err(e) => { /* Handle error... */ return Err(AuthorizationError::InternalError(format!("Failed check global perm: {}", e)).into()); }
                        }
                    }
                }
            };


            if !has_permission {
                let context = match &collection_name_clone {
                    Some(name) if required_permission != Permission::ManagePermissions => format!("for collection '{}'", name),
                    _ => "for this global operation".to_string(),
                };
                log::warn!("Authorization DENIED for user '{}' requiring {:?} {}", username, required_permission, context);
                return Err(AuthorizationError::Unauthorized(format!(
                    "User '{}' doesn't have required permissions {}",
                    username, context
                )).into());
            }

             log::debug!("Authorization GRANTED for user '{}' requiring {:?} (Collection: {:?})",
                username, required_permission, collection_name_clone);
            let res = fut.await?;
            Ok(res)
        })
    }
}

impl<S> AuthorizationMiddlewareService<S> {
    // Extract collection name from the request
    fn extract_collection_name(&self, req: &ServiceRequest) -> Result<Option<String>, AuthorizationError> {
        if let Some(path_params) = req.match_info().get("collection_id") {
            return Ok(Some(path_params.to_string()));
        }
        let path = req.path();
        if path.contains("/collections/") {
            let parts: Vec<&str> = path.split('/').collect();
            if let Some(idx) = parts.iter().position(|&r| r == "collections") {
                if idx + 1 < parts.len() {
                    return Ok(Some(parts[idx + 1].to_string()));
                }
            }
        }

        Ok(None)
    }
}


// Helper function to check global permissions by checking against the _system collection
fn check_global_permission(
    rbac_store: &RbacStore,
    username: &str,
    permission: Permission
) -> Result<bool, WaCustomError> {
    let system_collection_name = "_system";
    rbac_store.check_permission(username, system_collection_name, permission)
}
