use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[derive(Debug)]
pub enum VersionError {
    CollectionNotFound,
    InvalidVersionHash,
    UpdateFailed(String),
    DatabaseError(String),
}

impl Display for VersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CollectionNotFound => write!(f, "Collection not found"),
            Self::InvalidVersionHash => write!(f, "Invalid version hash"),
            Self::UpdateFailed(msg) => write!(f, "Failed to update version: {}", msg),
            Self::DatabaseError(msg) => write!(f, "Database error: {}", msg),
        }
    }
}

impl ResponseError for VersionError {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::html())
            .body(self.to_string())
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::CollectionNotFound => StatusCode::NOT_FOUND,
            Self::InvalidVersionHash => StatusCode::BAD_REQUEST,
            Self::UpdateFailed(_) => StatusCode::BAD_REQUEST,
            Self::DatabaseError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

