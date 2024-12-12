use super::types::{MetricResult, VectorId};
use crate::models::user::{AddUserResp, AuthResp, Statistics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub nn_count: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct FetchNeighbors {
    pub vector_db_name: String,
    pub vector_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct UpsertVectors {
    pub vector_db_name: String,
    pub vectors: Vec<Vector>,
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
        vector: Vector,
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
pub struct Vector {
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
