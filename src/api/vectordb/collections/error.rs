use crate::models::common::WaCustomError;
use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[allow(dead_code)]
#[derive(Debug)]
pub enum CollectionsError {
    NotFound,
    AlreadyExists(String),
    FailedToGetAppEnv,
    FailedToCreateCollection(String),
    WaCustomError(WaCustomError),
    ServerError(String),
}

impl Display for CollectionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectionsError::NotFound => write!(f, "Collection Not Found!"),
            CollectionsError::AlreadyExists(name) => write!(f, "Collection with name '{}' already exists", name),
            CollectionsError::FailedToGetAppEnv => write!(f, "Failed to get App Env!"),
            CollectionsError::FailedToCreateCollection(msg) => {
                write!(f, "Failed to create collection due to {}", msg)
            }
            CollectionsError::WaCustomError(e) => write!(f, "LMDB database error: {e:?}"),
            CollectionsError::ServerError(e) => write!(f, "Server error: {e}"),
        }
    }
}

impl ResponseError for CollectionsError {
    fn error_response(&self) -> actix_web::HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::html())
            .body(self.to_string())
    }
    fn status_code(&self) -> StatusCode {
        match self {
            CollectionsError::NotFound => StatusCode::BAD_REQUEST,
            CollectionsError::AlreadyExists(_) => StatusCode::CONFLICT,
            CollectionsError::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR,
            CollectionsError::FailedToCreateCollection(_) => StatusCode::BAD_REQUEST,
            CollectionsError::WaCustomError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            CollectionsError::ServerError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
