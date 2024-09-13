use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use std::fmt::Display;

#[derive(Debug)]
pub(crate) enum VectorsError {
    NotFound,
    FailedToGetAppEnv,
    FailedToCreateVector(String),
    NotImplemented,
    InternalServerError,
}

impl Display for VectorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorsError::NotFound => write!(f, "Vector Not Found!"),
            VectorsError::FailedToGetAppEnv => write!(f, "Failed to get App Env!"),
            VectorsError::FailedToCreateVector(msg) => {
                write!(f, "Failed to create vector due to {}", msg)
            }
            VectorsError::NotImplemented => {
                write!(f, "This is not supported yet!")
            }
            VectorsError::InternalServerError => {
                write!(f, "internal server error while trying to fetch vector!")
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
            VectorsError::NotFound => StatusCode::BAD_REQUEST,
            VectorsError::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR,
            VectorsError::FailedToCreateVector(_) => StatusCode::BAD_REQUEST,
            VectorsError::NotImplemented => StatusCode::BAD_REQUEST,
            VectorsError::InternalServerError => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
