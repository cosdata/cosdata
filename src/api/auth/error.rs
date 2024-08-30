use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[derive(Debug)]
pub(crate) enum AuthError {
    WrongCredentials,
    FailedToEncodeToken,
    InvalidToken,
}

impl Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::WrongCredentials => write!(f, "Wrong Credentials!"),
            AuthError::FailedToEncodeToken => write!(f, "failed to generate an jwt auth token!"),
            AuthError::InvalidToken => write!(f, "Invalid auth token!"),
        }
    }
}

impl ResponseError for AuthError {
    fn error_response(&self) -> actix_web::HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::html())
            .body(self.to_string())
    }
    fn status_code(&self) -> StatusCode {
        match self {
            AuthError::WrongCredentials => StatusCode::BAD_REQUEST,
            AuthError::FailedToEncodeToken => StatusCode::INTERNAL_SERVER_ERROR,
            AuthError::InvalidToken => StatusCode::UNAUTHORIZED,
        }
    }
}
