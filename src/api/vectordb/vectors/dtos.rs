use std::fmt;

use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize,
};

use crate::{
    indexes::inverted_index_types::SparsePair,
    models::{rpc::Vector, types::VectorId},
};

#[derive(Deserialize, Serialize, Debug)]
pub(crate) struct CreateDenseVectorDto {
    pub id: VectorId,
    pub values: Vec<f32>,
}

#[derive(Serialize, Debug)]
pub(crate) struct CreateSparseVectorDto {
    pub id: VectorId,
    pub values: Vec<SparsePair>,
}

impl<'de> Deserialize<'de> for CreateSparseVectorDto {
    fn deserialize<D>(deserializer: D) -> Result<CreateSparseVectorDto, D::Error>
    where
        D: Deserializer<'de>,
    {
        // A custom visitor to process the deserialization
        struct CreateSparseVectorDtoVisitor;

        impl<'de> Visitor<'de> for CreateSparseVectorDtoVisitor {
            type Value = CreateSparseVectorDto;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "struct CreateSparseVectorDto")
            }

            fn visit_map<M>(self, mut map: M) -> Result<CreateSparseVectorDto, M::Error>
            where
                M: de::MapAccess<'de>,
            {
                let mut id = None;
                let mut values: Option<Vec<f32>> = None;
                let mut indices: Option<Vec<u32>> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "id" => {
                            if id.is_some() {
                                return Err(de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        "values" => {
                            if values.is_some() {
                                return Err(de::Error::duplicate_field("values"));
                            }
                            values = Some(map.next_value()?);
                        }
                        "indices" => {
                            if indices.is_some() {
                                return Err(de::Error::duplicate_field("indices"));
                            }
                            indices = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(de::Error::unknown_field(
                                key.as_str(),
                                &["id", "values", "indices"],
                            ));
                        }
                    }
                }

                let id = id.ok_or_else(|| de::Error::missing_field("id"))?;
                let values = values.ok_or_else(|| de::Error::missing_field("values"))?;
                let indices = indices.ok_or_else(|| de::Error::missing_field("indices"))?;

                // Combine the values and indices into a Vec<SparsePair>
                let values = indices
                    .into_iter()
                    .zip(values.into_iter())
                    .map(|(index, value)| SparsePair(index, value))
                    .collect();

                Ok(CreateSparseVectorDto { id, values })
            }
        }

        deserializer.deserialize_map(CreateSparseVectorDtoVisitor)
    }
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CreateVectorDto {
    Dense(CreateDenseVectorDto),
    Sparse(CreateSparseVectorDto),
}

#[derive(Serialize)]
pub(crate) enum CreateVectorResponseDto {
    Dense(CreateDenseVectorDto),
    Sparse(CreateSparseVectorDto),
}

#[derive(Deserialize)]
pub(crate) struct UpdateVectorDto {
    pub values: Vec<f32>,
}

#[derive(Serialize)]
pub(crate) struct UpdateVectorResponseDto {
    pub id: VectorId,
    pub values: Vec<f32>,
    // pub created_at: String
}

#[derive(Deserialize)]
pub(crate) struct FindSimilarVectorsDto {
    pub vector: Vec<f32>,
    pub k: VectorId,
}

#[derive(Serialize)]
pub(crate) struct SimilarVector {
    pub id: VectorId,
    pub score: f32,
}

#[derive(Serialize)]
pub(crate) struct FindSimilarVectorsResponseDto {
    pub results: Vec<SimilarVector>,
}

#[derive(Deserialize)]
pub(crate) struct UpsertDto {
    pub vectors: Vec<Vector>,
}
