use crate::models::common::WaCustomError;
use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[allow(dead_code)]
#[derive(Debug)]
pub enum IndexesError {
    NotFound(String),
    FailedToGetAppEnv,
    CollectionNotFound,
    FailedToCreateIndex(String),
    IndexAlreadyExists(String),
    FailedToDeleteIndex(String),
    InvalidIndexType(String),
    WaCustom(WaCustomError),
}

impl Display for IndexesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "{}", msg),
            Self::FailedToGetAppEnv => write!(
                f,
                "Internal server error: Failed to get application environment"
            ),
            Self::CollectionNotFound => write!(f, "Collection not found"),
            Self::FailedToCreateIndex(msg) => write!(f, "Failed to create index: {}", msg),
            Self::IndexAlreadyExists(index_type) => write!(
                f,
                "Index of type '{}' already exists for this collection. Delete it first.",
                index_type
            ),
            Self::FailedToDeleteIndex(msg) => write!(f, "Failed to delete index: {}", msg),
            Self::InvalidIndexType(provided_type) => write!(
                f,
                "Invalid index type provided: '{}'. Expected 'dense' or 'sparse'.",
                provided_type
            ),
            Self::WaCustom(e) => write!(f, "Index operation failed due to internal error: {:?}", e),
        }
    }
}

impl ResponseError for IndexesError {
    fn error_response(&self) -> actix_web::HttpResponse {
        let status = self.status_code();
        let message = self.to_string();
        HttpResponse::build(status)
            .insert_header(ContentType::json())
            .json(serde_json::json!({
                "error": status.canonical_reason().unwrap_or("Error"),
                "code": status.as_u16(),
                "message": message
            }))
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR,
            Self::CollectionNotFound => StatusCode::NOT_FOUND,
            Self::FailedToCreateIndex(_) => StatusCode::BAD_REQUEST,
            Self::IndexAlreadyExists(_) => StatusCode::CONFLICT,
            Self::FailedToDeleteIndex(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InvalidIndexType(_) => StatusCode::BAD_REQUEST,
            Self::WaCustom(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl From<WaCustomError> for IndexesError {
    fn from(err: WaCustomError) -> Self {
        match err {
            WaCustomError::NotFound(msg) => IndexesError::NotFound(msg),
            e => IndexesError::WaCustom(e),
        }
    }
}
