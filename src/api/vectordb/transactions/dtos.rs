use crate::api::vectordb::vectors::dtos::CreateVectorDto;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, ToSchema)]
pub struct CreateTransactionResponseDto {
    pub transaction_id: String,
    #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
    pub created_at: DateTime<Utc>,
}

#[derive(Deserialize, ToSchema)]
pub struct UpsertDto {
    pub vectors: Vec<CreateVectorDto>,
}
