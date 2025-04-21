use crate::api::vectordb::vectors::dtos::CreateVectorDto;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub(crate) struct CreateTransactionResponseDto {
    pub transaction_id: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Deserialize)]
pub struct UpsertDto {
    pub vectors: Vec<CreateVectorDto>,
}
