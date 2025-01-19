use super::types::{MetricResult, VectorId};
use crate::{
    indexes::inverted_index_types::SparsePair,
    models::user::{AddUserResp, AuthResp, Statistics},
};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::{collections::HashMap, fmt};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Authenticate {
    username: String,
    password: String,
    pretty_print: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct AddUser {
    username: String,
    api_expiry_time: Option<String>,
    api_quota: Option<i32>,
    first_name: String,
    last_name: String,
    email: String,
    roles: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorANN {
    pub vector_db_name: String,
    pub vector: Vec<f32>,
    pub filter: Option<Filter>,
    pub nn_count: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct BatchVectorANN {
    pub vector_db_name: String,
    pub vectors: Vec<Vec<f32>>,
    pub filter: Option<Filter>,
    pub nn_count: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct FetchNeighbors {
    pub vector_db_name: String,
    pub vector_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpsertVectors {
    pub vector_db_name: String,
    pub vectors: Vec<DenseVector>,
}

#[derive(Deserialize)]
pub struct UpsertSparseVectors {
    pub vector_db_name: String,
    pub vectors: Vec<CreateSparseVectorDto>,
}

pub struct CreateSparseVectorDto {
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

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct CreateVectorDb {
    pub vector_db_name: String,
    pub dimensions: i32,
    pub max_val: Option<f32>,
    pub min_val: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RPCResponseBody {
    AuthenticateResp {
        auth: AuthResp,
    },
    RespAddUser {
        add_user: AddUserResp,
    },
    RespUpsertVectors {
        insert_stats: Option<Statistics>,
    },
    RespVectorKNN {
        knn: Vec<(u64, MetricResult)>,
    },
    RespFetchNeighbors {
        vector: DenseVector,
        neighbors: Vec<(u64, MetricResult)>,
    },
    #[serde(untagged)]
    RespCreateVectorDb {
        id: String,
        name: String,
        dimensions: usize,
        min_val: Option<f32>,
        max_val: Option<f32>,
        // created_at: String, // will be added when vector store has a creation timestamp
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseVector {
    pub id: VectorId,
    pub values: Vec<f32>,
}

// #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
// pub struct VectorList {
//     pub vectors: Vec<Vector>,
// }

pub type Single = MetadataColumnValue;
pub type Multiple = Vec<MetadataColumnValue>;

// Define the generic MetadataColumn type
#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum MetadataColumnValue {
    StringValue(String),
    IntValue(i32),
    FloatValue(f64),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum ComparisonOperator {
    #[serde(rename = "$eq")]
    Eq(Single),

    #[serde(rename = "$ne")]
    Ne(Single),

    #[serde(rename = "$gt")]
    Gt(Single),

    #[serde(rename = "$gte")]
    Gte(Single),

    #[serde(rename = "$lt")]
    Lt(Single),

    #[serde(rename = "$lte")]
    Lte(Single),

    #[serde(rename = "$in")]
    In(Multiple),

    #[serde(rename = "$nin")]
    Nin(Multiple),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum LogicalOperator {
    #[serde(rename = "$and")]
    And(Vec<Filter>),

    #[serde(rename = "$or")]
    Or(Vec<Filter>),
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum Filter {
    Comparison {
        #[serde(flatten)]
        column: HashMap<String, ComparisonOperator>,
    },
    Logical(LogicalOperator),
}
