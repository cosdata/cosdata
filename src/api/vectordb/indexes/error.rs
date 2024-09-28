use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

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
            IndexesError::NotFound => write!(f, "Index Not Found!"),
            IndexesError::FailedToGetAppEnv => write!(f, "Failed to get App Env!"),
            IndexesError::CollectionNotFound => write!(f, "Collection Not Found!"),
            IndexesError::FailedToCreateIndex(msg) => {
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
            IndexesError::NotFound => StatusCode::BAD_REQUEST,
            IndexesError::CollectionNotFound => StatusCode::BAD_REQUEST,
            IndexesError::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR,
            IndexesError::FailedToCreateIndex(_) => StatusCode::BAD_REQUEST,
        }
    }
}
