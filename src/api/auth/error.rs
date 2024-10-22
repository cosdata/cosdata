use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[derive(Debug)]
pub enum AuthError {
    WrongCredentials,
    FailedToEncodeToken,
    InvalidToken,
    FailedToExtractTokenFromRequest,
}

impl Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongCredentials => write!(f, "Wrong Credentials!"),
            Self::FailedToEncodeToken => write!(f, "failed to generate an jwt auth token!"),
            Self::InvalidToken => write!(f, "Invalid auth token!"),
            Self::FailedToExtractTokenFromRequest => {
                write!(f, "Failed to extract token from request!")
            }
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
            Self::WrongCredentials => StatusCode::BAD_REQUEST,
            Self::FailedToEncodeToken => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InvalidToken => StatusCode::UNAUTHORIZED,
            Self::FailedToExtractTokenFromRequest => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
