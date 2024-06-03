use crate::models::user::{AddUserResp, AuthResp, Statistics, User};
use serde::{Deserialize, Serialize};
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
    vector_db_name: String,
    vector: Vec<f32>,
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
    RespVectorKNN { knn: Vec<(i8, i8, String, f64)> },
    RespCreateVectorDb { result: bool },
}
