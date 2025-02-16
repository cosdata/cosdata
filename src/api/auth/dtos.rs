use actix_web::{dev::Payload, FromRequest, HttpMessage, HttpRequest};
use serde::{Deserialize, Serialize};

use super::error::AuthError;
use futures_util::future::{err, ok, Ready};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct CreateSessionDTO {
    pub username: String,
    pub password: String,
}

#[derive(Serialize)]
pub(crate) struct Session {
    pub access_token: String,
    pub created_at: u64,
    pub expires_at: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
// a structure for holding claims data used in JWT tokens
// resembles payload in NodeJS world
pub struct Claims {
    pub exp: u64,         // Expiry time of the token
    pub iat: u64,         // Issued at time of the token
    pub username: String, // Email associated with the token
}

impl FromRequest for Claims {
    type Error = AuthError;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        // Extract the token `Claims` from the request's extensions
        let extensions = req.extensions();
        match extensions.get::<Claims>() {
            Some(claims) => ok(claims.clone()),
            None => err(AuthError::FailedToExtractTokenFromRequest),
        }
    }
}
