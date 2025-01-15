use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

use crate::WaCustomError;

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum VectorsError {
    NotFound,
    FailedToGetAppEnv,
    FailedToCreateVector(String),
    FailedToUpdateVector(String),
    FailedToFindSimilarVectors(String),
    FailedToDeleteVector(String),
    NotImplemented,
    DatabaseError(String),
    InternalServerError,
    WaCustom(WaCustomError),
}

impl Display for VectorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Vector Not Found!"),
            Self::FailedToGetAppEnv => write!(f, "Failed to get App Env!"),
            Self::FailedToCreateVector(msg) => {
                write!(f, "Failed to create vector due to: {}", msg)
            }
            Self::NotImplemented => {
                write!(f, "This is not supported yet!")
            }
            Self::DatabaseError(msg) => write!(f, "Failed to fetch vector due to: {}", msg),
            Self::InternalServerError => {
                write!(f, "Internal server error while trying to fetch vector!")
            }
            Self::FailedToUpdateVector(msg) => {
                write!(f, "Failed to update vector due to: {}", msg)
            }
            Self::FailedToFindSimilarVectors(msg) => {
                write!(f, "Failed to find similar vectors due to: {}", msg)
            }
            Self::FailedToDeleteVector(msg) => {
                write!(f, "Failed to delete vector due to: {}", msg)
            }
            Self::WaCustom(e) => {
                write!(f, "Vector operation failed due to internal error: {e:?}")
            }
        }
    }
}

impl ResponseError for VectorsError {
    fn error_response(&self) -> actix_web::HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::html())
            .body(self.to_string())
    }
    fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::BAD_REQUEST,
            Self::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR,
            Self::FailedToCreateVector(_) => StatusCode::BAD_REQUEST,
            Self::NotImplemented => StatusCode::BAD_REQUEST,
            Self::DatabaseError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InternalServerError => StatusCode::INTERNAL_SERVER_ERROR,
            Self::FailedToUpdateVector(_) => StatusCode::BAD_REQUEST,
            Self::FailedToFindSimilarVectors(_) => StatusCode::BAD_REQUEST,
            Self::FailedToDeleteVector(_) => StatusCode::BAD_REQUEST,
            Self::WaCustom(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
