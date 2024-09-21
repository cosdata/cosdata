use chrono::{DateTime, Utc};
use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct CreateTransactionResponseDto {
    pub transaction_id: String,
    pub created_at: DateTime<Utc>,
}
