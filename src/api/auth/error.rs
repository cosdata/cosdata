use actix_web::{http::header::ContentType, HttpResponse, ResponseError};
use std::fmt::Display;

#[derive(Debug)]
pub(crate) enum AuthError {
    WrongCredentials,
}

impl Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::WrongCredentials => write!(f, "Wrong Credentials!"),
        }
    }
}

impl ResponseError for AuthError {
    fn error_response(&self) -> actix_web::HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::html())
            .body(self.to_string())
    }
    fn status_code(&self) -> actix_web::http::StatusCode {
        match self {
            AuthError::WrongCredentials => actix_web::http::StatusCode::BAD_REQUEST,
        }
    }
}
