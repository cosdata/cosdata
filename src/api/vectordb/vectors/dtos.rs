use std::fmt;

use crate::{
    metadata::MetadataFields,
    models::{collection::RawVectorEmbedding, types::DocumentId},
};

use serde::{
    de::{self, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};

use crate::{indexes::inverted::types::SparsePair, models::types::VectorId};

#[derive(Deserialize)]
pub struct VectorsQueryDto {
    pub document_id: DocumentId,
}

#[derive(Serialize)]
pub(crate) struct CreateVectorDto {
    pub id: VectorId,
    pub document_id: Option<DocumentId>,
    pub dense_values: Option<Vec<f32>>,
    pub metadata: Option<MetadataFields>,
    pub sparse_values: Option<Vec<SparsePair>>,
    pub text: Option<String>,
}

impl From<CreateVectorDto> for RawVectorEmbedding {
    fn from(dto: CreateVectorDto) -> Self {
        Self {
            id: dto.id,
            document_id: dto.document_id,
            dense_values: dto.dense_values,
            metadata: dto.metadata,
            sparse_values: dto.sparse_values,
            text: dto.text,
        }
    }
}

impl From<RawVectorEmbedding> for CreateVectorDto {
    fn from(emb: RawVectorEmbedding) -> Self {
        Self {
            id: emb.id,
            document_id: emb.document_id,
            dense_values: emb.dense_values,
            metadata: emb.metadata,
            sparse_values: emb.sparse_values,
            text: emb.text,
        }
    }
}

impl<'de> Deserialize<'de> for CreateVectorDto {
    fn deserialize<D>(deserializer: D) -> Result<CreateVectorDto, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CreateVectorDtoVisitor;

        impl<'de> Visitor<'de> for CreateVectorDtoVisitor {
            type Value = CreateVectorDto;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(
                    formatter,
                    "a vector with dense_values (+metadata), sparse (indices + values), or text"
                )
            }

            fn visit_map<M>(self, mut map: M) -> Result<CreateVectorDto, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut id = None;
                let mut document_id = None;
                let mut dense_values = None;
                let mut metadata = None;
                let mut sparse_values_raw: Option<(Vec<u32>, Vec<f32>)> = None;
                let mut text = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "id" => {
                            if id.is_some() {
                                return Err(de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        "document_id" => {
                            if document_id.is_some() {
                                return Err(de::Error::duplicate_field("document_id"));
                            }
                            document_id = map.next_value()?;
                        }
                        "dense_values" => {
                            if dense_values.is_some() {
                                return Err(de::Error::duplicate_field("dense_values"));
                            }
                            dense_values = Some(map.next_value()?);
                        }
                        "metadata" => {
                            if metadata.is_some() {
                                return Err(de::Error::duplicate_field("metadata"));
                            }
                            metadata = map.next_value()?;
                        }
                        "sparse_values" => {
                            let values: Vec<f32> = map.next_value()?;
                            if let Some((indices, _)) = sparse_values_raw.take() {
                                sparse_values_raw = Some((indices, values));
                            } else {
                                sparse_values_raw = Some((Vec::new(), values));
                            }
                        }
                        "sparse_indices" => {
                            let indices: Vec<u32> = map.next_value()?;
                            if let Some((_, values)) = sparse_values_raw.take() {
                                sparse_values_raw = Some((indices, values));
                            } else {
                                sparse_values_raw = Some((indices, Vec::new()));
                            }
                        }
                        "text" => {
                            if text.is_some() {
                                return Err(de::Error::duplicate_field("text"));
                            }
                            text = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(de::Error::unknown_field(
                                &key,
                                &[
                                    "id",
                                    "document_id",
                                    "dense_values",
                                    "metadata",
                                    "sparse_values",
                                    "sparse_indices",
                                    "text",
                                ],
                            ));
                        }
                    }
                }

                let id = id.ok_or_else(|| de::Error::missing_field("id"))?;

                let sparse_values = match sparse_values_raw {
                    Some((indices, values)) => {
                        if indices.len() != values.len() {
                            return Err(de::Error::custom(
                                "length mismatch between sparse_indices and sprase_values",
                            ));
                        }
                        Some(
                            indices
                                .into_iter()
                                .zip(values)
                                .map(|(dim, val)| SparsePair(dim, val))
                                .collect(),
                        )
                    }
                    None => None,
                };

                Ok(CreateVectorDto {
                    id,
                    document_id,
                    dense_values,
                    metadata,
                    sparse_values,
                    text,
                })
            }
        }

        deserializer.deserialize_map(CreateVectorDtoVisitor)
    }
}

#[derive(Serialize)]
pub(crate) struct SimilarVector {
    pub id: VectorId,
    pub score: f32,
}
