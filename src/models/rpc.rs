use super::types::VectorHash;
use crate::models::user::{AddUserResp, AuthResp, Statistics, User};
use rayon::iter::WhileSome;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

pub struct UpsertVectors {
    pub vector_db_name: String,
    pub vector: Vec<Vec<f32>>,
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
    AuthenticateResp { auth: AuthResp },
    RespAddUser { add_user: AddUserResp },
    RespUpsertVectors { insert_stats: Option<Statistics> },
    RespVectorKNN { knn: Option<Vec<(VectorHash, f32)>> },
    RespCreateVectorDb { result: bool },
}
pub type Single = MetadataColumnValue;
pub type Multiple = Vec<MetadataColumnValue>;

// Define the generic MetadataColumn type
#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
pub enum MetadataColumnValue {
    StringValue(String),
    IntValue(i32),
    FloatValue(f64),
    // Add other types as needed
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
    // Add other operators as needed
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum LogicalOperator {
    #[serde(rename = "$and")]
    And(Vec<Filter>),

    #[serde(rename = "$or")]
    Or(Vec<Filter>),
    // Add other logical operators as needed
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
