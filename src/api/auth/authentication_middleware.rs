use crate::models::crypto::get_current_timestamp;
use crate::models::types::SessionDetails;

use super::dtos::Claims;
use super::error::AuthError;

use std::future::{ready, Ready};
use std::sync::Arc;

use actix_web::http::header::{self};
use actix_web::HttpMessage;
use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error,
};
use dashmap::DashMap;
use futures_util::future::LocalBoxFuture;

// There are two steps in middleware processing.
// 1. Middleware initialization, middleware factory gets called with
//    next service in chain as parameter.
// 2. Middleware's call method gets called with normal request.
pub(crate) struct AuthenticationMiddleware(pub Arc<DashMap<String, SessionDetails>>);

// Middleware factory is `Transform` trait
// `S` - type of the next service
// `B` - type of response's body
impl<S, B> Transform<S, ServiceRequest> for AuthenticationMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = AuthenticationMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(AuthenticationMiddlewareService {
            service,
            active_sessions: self.0.clone(),
        }))
    }
}

pub(crate) struct AuthenticationMiddlewareService<S> {
    service: S,
    active_sessions: Arc<DashMap<String, SessionDetails>>,
}

impl<S, B> Service<ServiceRequest> for AuthenticationMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let claims = authentication_middleware(&req, &self.active_sessions);

        let claims = match claims {
            Ok(claims) => claims,
            Err(e) => return Box::pin(async move { Err(e.into()) }),
        };

        req.extensions_mut().insert(claims);

        let fut = self.service.call(req);
        Box::pin(async move {
            let res = fut.await?;
            Ok(res)
        })
    }
}

fn authentication_middleware(
    req: &ServiceRequest,
    active_sessions: &DashMap<String, SessionDetails>,
) -> Result<Claims, AuthError> {
    let auth_header = req.headers().get(header::AUTHORIZATION);
    let auth_header = auth_header.ok_or(AuthError::InvalidToken)?;
    let auth_header = auth_header.to_str().map_err(|_| AuthError::InvalidToken)?;

    let mut header = auth_header.split_whitespace();
    let (_, access_token) = (header.next(), header.next().ok_or(AuthError::InvalidToken)?);
    let session = active_sessions
        .get(access_token)
        .ok_or(AuthError::InvalidToken)?;
    let current_time = get_current_timestamp();
    if current_time >= session.expires_at {
        drop(session);
        active_sessions.remove(access_token);
        return Err(AuthError::InvalidToken);
    }
    let claims = Claims {
        iat: session.created_at,
        exp: session.expires_at,
        username: session.user.username.clone(),
    };
    Ok(claims)
}
