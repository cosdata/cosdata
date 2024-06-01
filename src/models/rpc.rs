use serde::{Deserialize, Serialize};
use crate::models::user::{AuthResp, AddUserResp, User, Statistics};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RPCReqParams {
    Authenticate {
        username: String,
        password: String,
        pretty_print: bool,
    },
    Request {
        session_key: String,
        pretty_print: bool,
        method_params: RPCReqMethodParams,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RPCReqMethodParams {
    AddUser {
        username: String,
        api_expiry_time: Option<String>,
        api_quota: Option<i32>,
        first_name: String,
        last_name: String,
        email: String,
        roles: Option<Vec<String>>,
    },
    VectorKNN {
        vector_db_name: String,
        vector: Vec<f32>,
    },
    UpsertVectors {
        vector_db_name: String,
        vector: Vec<Vec<f32>>,
    },
    CreateVectorDb {
        vector_db_name: String,
        dimensions: i32,
        max_val: Option<f32>,
        min_val: Option<f32>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RPCMessage {
    RPCRequest {
        rq_method: String,
        rq_params: RPCReqParams,
    },
    RPCResponse {
        rs_status_code: i16,
        pretty: bool,
        rs_resp: Result<Option<RPCResponseBody>, RPCError>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RPCError {
    pub rs_status_message: RPCErrors,
    pub rs_error_data: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RPCErrors {
    InvalidMethod,
    ParseError,
    InvalidParams,
    InternalError,
    ServerError,
    InvalidRequest,
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
        knn: Vec<(i8, i8, String, f64)>,
    },
    RespCreateVectorDb {
        result: bool,
    },
}
