use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};

use super::{
    dtos::{Claims, LoginCredentials},
    error::AuthError,
};

pub(crate) async fn login(credentials: LoginCredentials) -> Result<String, AuthError> {
    const USERNAME: &str = "admin";
    const PASSWORD: &str = "admin";

    if credentials.username == USERNAME && credentials.password == PASSWORD {
        encode_jwt(&credentials.username)
    } else {
        Err(AuthError::WrongCredentials)
    }
}

pub fn encode_jwt(username: &str) -> Result<String, AuthError> {
    let secret: String = "randomStringTypicallyFromEnv".to_string();
    let now = Utc::now();
    let expire = Duration::hours(24);
    let exp = now + expire;
    let iat = now;

    let claim = Claims {
        iat: iat.timestamp(),
        exp: exp.timestamp(),
        username: username.to_string(),
    };

    encode(
        &Header::default(),
        &claim,
        &EncodingKey::from_secret(secret.as_ref()),
    )
    .map_err(|_| AuthError::FailedToEncodeToken)
}

pub fn decode_jwt(jwt_token: String) -> Result<TokenData<Claims>, AuthError> {
    let secret = "randomStringTypicallyFromEnv".to_string();

    let result: Result<TokenData<Claims>, AuthError> = decode(
        &jwt_token,
        &DecodingKey::from_secret(secret.as_ref()),
        &Validation::default(),
    )
    .map_err(|_| AuthError::InvalidToken);
    result
}
