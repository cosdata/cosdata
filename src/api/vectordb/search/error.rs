use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

use crate::models::common::WaCustomError;

#[derive(Debug)]
pub(crate) enum SearchError {
    CollectionNotFound(String),
    IndexNotFound(String),
    InvalidFilter(String),
    InternalServerError(String),
    WaCustom(WaCustomError),
    #[allow(dead_code)]
    InvalidInput(String),
}

impl Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::CollectionNotFound(name) => write!(f, "Collection '{}' not found", name),
            SearchError::IndexNotFound(msg) => write!(f, "Required index not found: {}", msg),
            SearchError::InvalidFilter(msg) => write!(f, "Invalid metadata filter: {}", msg),
            SearchError::InternalServerError(msg) => write!(f, "Internal server error: {}", msg),
            SearchError::WaCustom(e) => write!(f, "Internal search error: {:?}", e),
            Self::InvalidInput(msg) => write!(f, "Invalid input for search: {}", msg),
        }
    }
}

impl ResponseError for SearchError {
    fn error_response(&self) -> HttpResponse {
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
            SearchError::CollectionNotFound(_) => StatusCode::NOT_FOUND,
            SearchError::IndexNotFound(_) => StatusCode::BAD_REQUEST,
            SearchError::InvalidFilter(_) => StatusCode::BAD_REQUEST,
            SearchError::InternalServerError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            SearchError::WaCustom(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InvalidInput(_) => StatusCode::BAD_REQUEST,
        }
    }
}

impl From<WaCustomError> for SearchError {
    fn from(err: WaCustomError) -> Self {
        match err {
            WaCustomError::NotFound(msg) => SearchError::CollectionNotFound(msg),
            WaCustomError::MetadataError(e) => SearchError::InvalidFilter(e.to_string()),
            e => SearchError::WaCustom(e),
        }
    }
}
