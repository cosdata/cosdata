use std::fmt::Display;

use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};

#[derive(Debug)]
pub(crate) enum TransactionError {
    NotFound,
    CollectionNotFound,
    OnGoingTransaction,
    FailedToGetAppEnv,
    FailedToCreateTransaction(String),
    NotImplemented,
}

impl Display for TransactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransactionError::NotFound => write!(f, "Transaction Not Found!"),
            TransactionError::CollectionNotFound => write!(f, "Collection not found!"),
            TransactionError::FailedToGetAppEnv => write!(f, "Failed to get App Env!"),
            TransactionError::OnGoingTransaction => write!(f, "There is an on-going transaction!"),
            TransactionError::FailedToCreateTransaction(msg) => {
                write!(f, "Failed to create transaction due to {}", msg)
            }
            TransactionError::NotImplemented => {
                write!(f, "This is not supported yet!")
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
            TransactionError::NotFound => StatusCode::BAD_REQUEST,
            TransactionError::CollectionNotFound => StatusCode::BAD_REQUEST,
            TransactionError::FailedToGetAppEnv => StatusCode::INTERNAL_SERVER_ERROR,
            TransactionError::FailedToCreateTransaction(_) => StatusCode::BAD_REQUEST,
            TransactionError::OnGoingTransaction => StatusCode::CONFLICT,
            TransactionError::NotImplemented => StatusCode::BAD_REQUEST,
        }
    }
}
