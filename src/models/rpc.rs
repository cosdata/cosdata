use serde::{Deserialize, Serialize};
use crate::models::user::{AuthResp, AddUserResp, User, Statistics};
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RPCReqParams {
    AuthenticateReq {
        username: String,
        password: String,
        pretty_print: bool,
    },
    GeneralReq {
        session_key: String,
        pretty_print: bool,
        method_params: Option<RPCReqMethodParams>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RPCReqMethodParams {
    AddUser {
        au_username: String,
        au_api_expiry_time: Option<String>,
        au_api_quota: Option<i32>,
        au_first_name: String,
        au_last_name: String,
        au_email: String,
        au_roles: Option<Vec<String>>,
    },
    VectorKNN {
        knn_vector_db_name: String,
        knn_vector: Vec<f32>,
    },
    UpsertVectors {
        uv_vector_db_name: String,
        uv_vector: Vec<Vec<f32>>,
    },
    CreateVectorDb {
        cv_vector_db_name: String,
        cv_dimensions: i32,
        cv_max_val: Option<f32>,
        cv_min_val: Option<f32>,
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
