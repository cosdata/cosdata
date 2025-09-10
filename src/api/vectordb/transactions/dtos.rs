use crate::{
    api::vectordb::vectors::dtos::CreateVectorDto,
    models::{collection::OmVectorEmbedding, collection_transaction::ExplicitTransactionID},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, ToSchema)]
pub struct CreateTransactionResponseDto {
    pub transaction_id: ExplicitTransactionID,
    #[schema(value_type = String, example = "2023-01-01T12:00:00Z")]
    pub created_at: DateTime<Utc>,
}

#[derive(Deserialize, ToSchema)]
pub struct UpsertDto {
    pub vectors: Vec<CreateVectorDto>,
}

#[derive(Deserialize, ToSchema)]
pub struct OmUpsertDto {
    pub vectors: Vec<OmVectorEmbedding>,
}
