use crate::{
    api::vectordb::{indexes::dtos::IndexType, vectors::dtos::CreateSparseVectorDto},
    models::rpc::DenseVector,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub(crate) struct CreateTransactionDto {
    pub index_type: IndexType,
}

#[derive(Deserialize)]
pub(crate) struct CommitTransactionDto {
    pub index_type: IndexType,
}

#[derive(Deserialize)]
pub(crate) struct AbortTransactionDto {
    pub index_type: IndexType,
}

#[derive(Serialize)]
pub(crate) struct CreateTransactionResponseDto {
    pub transaction_id: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Deserialize)]
#[serde(tag = "index_type", content = "vectors", rename_all = "lowercase")]
pub enum UpsertDto {
    Dense(Vec<DenseVector>),
    Sparse(Vec<CreateSparseVectorDto>),
}
