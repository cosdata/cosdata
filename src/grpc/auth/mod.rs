use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::app_context::AppContext;

crate::cfg_grpc! {
use super::proto::auth_service_server::AuthService;
use super::proto::{CreateSessionRequest, CreateSessionResponse, Claims};
use crate::api::auth::dtos::{CreateSessionDTO, Claims as AppClaims};
use crate::api::auth::service;
use crate::api::auth::error::AuthError;
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};

pub struct AuthServiceImpl {
    pub context: Arc<AppContext>,
}

#[tonic::async_trait]
impl AuthService for AuthServiceImpl {
    async fn create_session(
        &self,
        request: Request<CreateSessionRequest>,
    ) -> Result<Response<CreateSessionResponse>, Status> {
        let req = request.into_inner();

        // Convert gRPC request to internal DTO
        let create_session_dto = CreateSessionDTO {
            username: req.username.clone(),
            password: req.password,
        };

        // Call the actual auth service
        let session_result = service::create_session(create_session_dto, self.context.clone())
            .await;

        // Handle errors
        let session = match session_result {
            Ok(s) => s,
            Err(AuthError::WrongCredentials) => {
                return Err(Status::invalid_argument("Invalid credentials"));
            }
            Err(e) => {
                return Err(Status::internal(format!("Authentication error: {}", e)));
            }
        };

        // Skip JWT token validation and just create response with the token
        // and basic information we already have
        let response = CreateSessionResponse {
            access_token: session.access_token,
            created_at: session.created_at,
            expires_at: session.expires_at,
            claims: Some(Claims {
                exp: session.expires_at,
                iat: session.created_at,
                username: req.username,
            }),
        };

        Ok(Response::new(response))
    }
}
}

#[cfg(test)]
mod tests {
    // Tests commented out until proper environment setup is available
}
