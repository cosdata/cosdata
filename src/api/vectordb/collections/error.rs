use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[derive(Debug)]
pub enum CollectionsError {
    NotFound,
    FailedToGetAppEnv
}

impl Display for CollectionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectionsError::NotFound => write!(f, "Collection Not Found!"),
            CollectionsError::FailedToGetAppEnv => write!(f, "Failed to get App Env!")
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
            CollectionsError::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}
