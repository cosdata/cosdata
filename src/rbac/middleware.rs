use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    error::Error as ActixError,
    http::StatusCode,
    web, HttpMessage, HttpResponse,
};
use futures_util::future::LocalBoxFuture;
use futures_util::future::{ready, Ready};
use std::sync::Arc;

use crate::api::auth::dtos::Claims;
use crate::app_context::AppContext;
use crate::models::common::WaCustomError;
use crate::rbac::store::RbacStore;
use crate::rbac::Permission;

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
            Self::MissingCollection => write!(f, "Missing or invalid collection identifier in the request path for a permission that requires it."),
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

#[derive(Clone, Copy)]
pub struct AuthorizationMiddleware {
    pub required_permission: Permission,
}

impl<S, B> Transform<S, ServiceRequest> for AuthorizationMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = ActixError;
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
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = ActixError;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        log::info!(
            "[RBAC FOR ROUTE] Path: {} - From Middleware Wrapper for CreateIndex (if this prints)",
            req.path()
        );
        log::info!(
            "[RBAC Middleware] Attempting to authorize path: {}, Required Permission: {:?}",
            req.path(),
            self.required_permission
        );
        let claims = match req.extensions().get::<Claims>() {
            Some(claims) => claims.clone(),
            None => {
                return Box::pin(async {
                    Err(
                        AuthorizationError::Unauthorized("User not authenticated".to_string())
                            .into(),
                    )
                });
            }
        };

        let app_ctx_data = match req.app_data::<web::Data<AppContext>>() {
            Some(ctx) => ctx.clone(),
            None => {
                log::error!("RBAC Middleware: AppContext not found in request data.");
                return Box::pin(async {
                    Err(
                        AuthorizationError::InternalError("App context not found".to_string())
                            .into(),
                    )
                });
            }
        };
        // Deref Data to get Arc<AppContext>
        let app_ctx: Arc<AppContext> = app_ctx_data.into_inner();
        let rbac_store = app_ctx.rbac_store.clone();

        let collection_name_option = self.extract_collection_name(&req);

        let username = claims.username.clone();
        let required_permission = self.required_permission;

        let collection_name_for_async = collection_name_option.clone();

        let service_call_fut = self.service.call(req);

        Box::pin(async move {
            let has_permission = if required_permission == Permission::ManagePermissions {
                log::debug!(
                    "Performing GLOBAL check for ManagePermissions for user '{}'",
                    username
                );
                match Self::check_global_permission(&rbac_store, &username, required_permission) {
                    Ok(result) => result,
                    Err(e) => {
                        log::error!(
                            "RBAC: Error during ManagePermissions check for user '{}': {}",
                            username,
                            e
                        );
                        return Err(AuthorizationError::InternalError(format!(
                            "Failed check global perm: {}",
                            e
                        ))
                        .into());
                    }
                }
            } else if required_permission == Permission::DeleteCollection {
                log::debug!(
                    "Performing GLOBAL check for DeleteCollection for user '{}'",
                    username
                );
                match Self::check_global_permission(&rbac_store, &username, required_permission) {
                    Ok(true) => {
                        log::debug!(
                            "GLOBAL DeleteCollection permission GRANTED for user '{}'",
                            username
                        );
                        true
                    }
                    Ok(false) => {
                        log::debug!("Global DeleteCollection check failed, checking specific collection for user '{}'...", username);
                        match &collection_name_for_async {
                            Some(collection) => {
                                log::debug!("Performing collection check for DeleteCollection on '{}' for user '{}'", collection, username);
                                match rbac_store.check_permission(
                                    &username,
                                    collection,
                                    required_permission,
                                ) {
                                    Ok(result) => result,
                                    Err(e) => {
                                        log::error!("RBAC: Error during specific DeleteCollection check for user '{}', collection '{}': {}", username, collection, e);
                                        return Err(AuthorizationError::InternalError(format!(
                                            "Failed check specific perm: {}",
                                            e
                                        ))
                                        .into());
                                    }
                                }
                            }
                            None => {
                                log::debug!("No specific collection provided for DeleteCollection check for user '{}'.", username);
                                false // If no global and no specific collection to check against, then false.
                            }
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "RBAC: Error during global DeleteCollection check for user '{}': {}",
                            username,
                            e
                        );
                        return Err(AuthorizationError::InternalError(format!(
                            "Failed check global perm: {}",
                            e
                        ))
                        .into());
                    }
                }
            } else {
                match &collection_name_for_async {
                    Some(collection) => {
                        // Collection-specific check
                        log::debug!("Performing collection check for permission '{:?}' on collection '{}' for user '{}'",
                            required_permission, collection, username);
                        match rbac_store.check_permission(
                            &username,
                            collection,
                            required_permission,
                        ) {
                            Ok(result) => result,
                            Err(e) => {
                                log::error!("RBAC: Error during specific permission check for user '{}', collection '{}', perm '{:?}': {}", username, collection, required_permission, e);
                                return Err(AuthorizationError::InternalError(format!(
                                    "Failed check specific perm: {}",
                                    e
                                ))
                                .into());
                            }
                        }
                    }
                    None => {
                        log::debug!("No specific collection. Performing GLOBAL check for permission '{:?}' for user '{}'",
                            required_permission, username);
                        match Self::check_global_permission(
                            &rbac_store,
                            &username,
                            required_permission,
                        ) {
                            Ok(result) => result,
                            Err(e) => {
                                log::error!("RBAC: Error during global check for user '{}', perm '{:?}': {}", username, required_permission, e);
                                return Err(AuthorizationError::InternalError(format!(
                                    "Failed check global perm: {}",
                                    e
                                ))
                                .into());
                            }
                        }
                    }
                }
            };

            if !has_permission {
                let context_msg = match &collection_name_for_async {
                    Some(name) => format!("for collection '{}'", name),
                    _ => "for this global operation".to_string(),
                };
                log::warn!(
                    "Authorization DENIED for user '{}' requiring {:?} {}",
                    username,
                    required_permission,
                    context_msg
                );
                return Err(AuthorizationError::Unauthorized(format!(
                    "User '{}' does not have permission {:?} {}",
                    username, required_permission, context_msg
                ))
                .into());
            }

            log::debug!(
                "Authorization GRANTED for user '{}' requiring {:?} (Collection: {:?})",
                username,
                required_permission,
                collection_name_for_async
            );

            let res = service_call_fut.await?;
            Ok(res)
        })
    }
}

impl<S> AuthorizationMiddlewareService<S> {
    fn extract_collection_name(&self, req: &ServiceRequest) -> Option<String> {
        if let Some(id) = req.match_info().get("collection_id") {
            if !id.is_empty() {
                return Some(id.to_string());
            }
        }
        if let Some(name) = req.match_info().get("collection_name") {
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }

        let path = req.path();
        if path.starts_with("/vectordb/collections/") {
            let parts: Vec<&str> = path.split('/').collect();

            if parts.len() > 3
                && parts[0].is_empty()
                && parts[1] == "vectordb"
                && parts[2] == "collections"
            {
                let potential_collection_name = parts[3];
                if !potential_collection_name.is_empty()
                    && req.match_info().get(potential_collection_name).is_none()
                    && potential_collection_name != "search"
                    && potential_collection_name != "indexes"
                    && potential_collection_name != "vectors"
                    && potential_collection_name != "transactions"
                    && potential_collection_name != "sync-transaction"
                    && potential_collection_name != "versions"
                    && potential_collection_name != "loaded"
                {
                    log::trace!("RBAC Middleware: Extracted '{}' as potential collection name from path segment.", potential_collection_name);
                    return Some(potential_collection_name.to_string());
                }
            }
        }
        None
    }

    fn check_global_permission(
        rbac_store: &Arc<RbacStore>,
        username: &str,
        permission: Permission,
    ) -> Result<bool, WaCustomError> {
        let system_collection_name = "_system";
        rbac_store.check_permission(username, system_collection_name, permission)
    }
}
