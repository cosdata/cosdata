use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[allow(dead_code)]
#[derive(Debug)]
pub enum IndexesError {
    NotFound,
    FailedToGetAppEnv,
    CollectionNotFound,
    FailedToCreateIndex(String),
}

impl Display for IndexesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Index Not Found!"),
            Self::FailedToGetAppEnv => write!(f, "Failed to get App Env!"),
            Self::CollectionNotFound => write!(f, "Collection Not Found!"),
            Self::FailedToCreateIndex(msg) => {
                write!(f, "Failed to create index due to {}", msg)
            }
        }
    }
}

impl ResponseError for IndexesError {
    fn error_response(&self) -> actix_web::HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::html())
            .body(self.to_string())
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::BAD_REQUEST,
            Self::CollectionNotFound => StatusCode::BAD_REQUEST,
            Self::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR,
            Self::FailedToCreateIndex(_) => StatusCode::BAD_REQUEST,
        }
    }
}
