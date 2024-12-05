use std::fmt::Display;

use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum TransactionError {
    NotFound,
    CollectionNotFound,
    OnGoingTransaction,
    FailedToGetAppEnv,
    FailedToCreateTransaction(String),
    FailedToCommitTransaction(String),
    FailedToCreateVector(String),
    FailedToDeleteVector(String),
    NotImplemented,
}

impl Display for TransactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Transaction Not Found!"),
            Self::CollectionNotFound => write!(f, "Collection not found!"),
            Self::FailedToGetAppEnv => write!(f, "Failed to get App Env!"),
            Self::OnGoingTransaction => write!(f, "There is an on-going transaction!"),
            Self::FailedToCreateTransaction(msg) => {
                write!(f, "Failed to create transaction due to {}", msg)
            }
            Self::FailedToCommitTransaction(msg) => {
                write!(f, "Failed to commit transaction due to {}", msg)
            }
            Self::NotImplemented => {
                write!(f, "This is not supported yet!")
            }
            Self::FailedToCreateVector(msg) => {
                write!(f, "Failed to create vector in transaction due to {}", msg)
            }
            Self::FailedToDeleteVector(msg) => {
                write!(f, "Failed to delete vector in transaction due to: {}", msg)
            }
        }
    }
}

impl ResponseError for TransactionError {
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
            Self::FailedToCreateTransaction(_) => StatusCode::BAD_REQUEST,
            Self::FailedToCommitTransaction(_) => StatusCode::BAD_REQUEST,
            Self::OnGoingTransaction => StatusCode::CONFLICT,
            Self::NotImplemented => StatusCode::BAD_REQUEST,
            Self::FailedToCreateVector(_) => StatusCode::BAD_REQUEST,
            Self::FailedToDeleteVector(_) => StatusCode::BAD_REQUEST,
        }
    }
}
