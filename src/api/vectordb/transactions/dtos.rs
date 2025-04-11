use crate::{
    api::vectordb::{
        indexes::dtos::IndexType,
        vectors::dtos::{CreateSparseIdfDocumentDto, CreateSparseVectorDto},
    },
    models::rpc::DenseVector,
};
use chrono::{DateTime, Utc};
use serde::{de, Deserialize, Deserializer, Serialize};

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

pub enum UpsertDto {
    Dense(Vec<DenseVector>),
    Sparse(Vec<CreateSparseVectorDto>),
    SparseIdf(Vec<CreateSparseIdfDocumentDto>),
}

impl<'de> Deserialize<'de> for UpsertDto {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TypeProbe {
            index_type: String,
            #[serde(default)]
            is_idf: bool,
        }

        // Deserialize to a raw value first
        let value = serde_json::Value::deserialize(deserializer)?;

        // Probe index_type and is_idf
        let probe: TypeProbe = serde_json::from_value(value.clone()).map_err(de::Error::custom)?;

        match (probe.index_type.as_str(), probe.is_idf) {
            ("dense", _) => {
                #[derive(Deserialize)]
                struct DenseWrapper {
                    vectors: Vec<DenseVector>,
                }
                let wrapper: DenseWrapper =
                    serde_json::from_value(value).map_err(de::Error::custom)?;
                Ok(UpsertDto::Dense(wrapper.vectors))
            }
            ("sparse", true) => {
                #[derive(Deserialize)]
                struct SparseIdfWrapper {
                    documents: Vec<CreateSparseIdfDocumentDto>,
                }
                let wrapper: SparseIdfWrapper =
                    serde_json::from_value(value).map_err(de::Error::custom)?;
                Ok(UpsertDto::SparseIdf(wrapper.documents))
            }
            ("sparse", false) => {
                #[derive(Deserialize)]
                struct SparseWrapper {
                    vectors: Vec<CreateSparseVectorDto>,
                }
                let wrapper: SparseWrapper =
                    serde_json::from_value(value).map_err(de::Error::custom)?;
                Ok(UpsertDto::Sparse(wrapper.vectors))
            }
            (other, _) => Err(de::Error::unknown_variant(other, &["dense", "sparse"])),
        }
    }
}
