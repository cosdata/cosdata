// mod api_data;
// use api_data::{AuthResp, RPCError, RPCErrors, RPCMessage, RPCReqMethodParams, RPCReqParams, RPCResponseBody, Statistics, XDataReq};
// use serde::{Deserialize, Serialize};
// use std::sync::{Arc, Mutex, RwLock};
// use log::info;

use crate::models::encoding_format::EncodingFormat;
use crate::models::rpc::{
    RPCReqParams, RPCReqMethodParams, RPCMessage, RPCError, RPCErrors, RPCResponseBody,
};
use crate::models::user::{AuthResp, AddUserResp, User, Statistics};

use std::sync::{Arc, Mutex};
use log::info;





async fn auth_login_client(msg: RPCMessage, pretty: bool) -> Result<RPCMessage, RPCError> {
    let dbe = get_db().await;
    match msg {
        RPCMessage::RPCRequest { rq_method, rq_params } => {
            if rq_method == "AUTHENTICATE" {
                match rq_params {
                    RPCReqParams::AuthenticateReq { username, password, pretty } => {
                        let resp = login(&username, &password).await;
                        Ok(RPCMessage::RPCResponse {
                            rs_status_code: 200,
                            pretty,
                            rs_resp: Ok(Some(RPCResponseBody::AuthenticateResp { auth: resp })),
                        })
                    }
                    _ => Ok(RPCMessage::RPCResponse {
                        rs_status_code: 404,
                        pretty,
                        rs_resp: Err(RPCError {
                            rs_status_message: RPCErrors::InvalidRequest,
                            rs_error_data: None,
                        }),
                    }),
                }
            } else {
                Ok(RPCMessage::RPCResponse {
                    rs_status_code: 200,
                    pretty,
                    rs_resp: Ok(Some(RPCResponseBody::AuthenticateResp {
                        auth: AuthResp {
                            session_key: None,
                            calls_used: 0,
                            calls_remaining: 0,
                        },
                    })),
                })
            }
        }
        _ => Err(RPCError {
            rs_status_message: RPCErrors::InvalidRequest,
            rs_error_data: None,
        }),
    }
}

async fn delegate_request(enc_req: RPCMessage) -> Result<RPCMessage, RPCError> {
    let dbe = get_db().await;
    let ain = get_ain().await;

    match enc_req {
        RPCMessage::RPCRequest { rq_params, .. } => match rq_params {
            RPCReqParams::AuthenticateReq { .. } => auth_login_client(enc_req, true).await,
            RPCReqParams::GeneralReq {
                session_key,
                pretty,
                ..
            } => {
                let user_data = lookup_user_data(&session_key).await;

                match user_data {
                    None => go_get_resource(enc_req, vec![], &session_key, pretty).await,
                    Some((name, quota, used, exp, roles)) => {
                        let curtm = get_current_time().await;
                        if exp > curtm && quota > used {
                            if (used + 1) % 100 == 0 {
                                // Placeholder for additional logic
                            }
                            update_user_data(&session_key, name, quota, used + 1, exp, roles).await;
                            go_get_resource(enc_req, roles, &session_key, pretty).await
                        } else {
                            delete_user_data(&session_key).await;
                            Ok(RPCMessage::RPCResponse {
                                rs_status_code: 200,
                                pretty,
                                rs_resp: Ok(Some(RPCResponseBody::AuthenticateResp {
                                    auth: AuthResp {
                                        session_key: None,
                                        calls_used: 0,
                                        calls_remaining: 0,
                                    },
                                })),
                            })
                        }
                    }
                }
            }
        },
        _ => Err(RPCError {
            rs_status_message: RPCErrors::InvalidRequest,
            rs_error_data: None,
        }),
    }
}

async fn go_get_resource(
    msg: RPCMessage,
    roles: Vec<String>,
    sess_key: &str,
    pretty: bool,
) -> Result<RPCMessage, RPCError> {
    let dbe = get_db().await;
    let ain = get_ain().await;

    match msg {
        RPCMessage::RPCRequest { rq_method, rq_params } => {
            match rq_method.as_str() {
                "ADD_USER" => {
                    // Placeholder for ADD_USER logic
                    unimplemented!()
                }
                "CREATE_VECTOR_DB" => {
                    info!("Creating Vector DB");
                    match rq_params {
                        RPCReqParams::GeneralReq {
                            method_params: Some(RPCReqMethodParams::CreateVectorDb {
                                cv_vector_db_name,
                                cv_dimensions,
                                cv_max_val,
                                cv_min_val,
                            }),
                            ..
                        } => {
                            info!("Creating Vector DB: {}", cv_vector_db_name);
                            init_vector_store(
                                &cv_vector_db_name,
                                cv_dimensions as usize,
                                cv_max_val,
                                cv_min_val,
                            )
                            .await;
                            Ok(RPCMessage::RPCResponse {
                                rs_status_code: 200,
                                pretty,
                                rs_resp: Ok(Some(RPCResponseBody::RespCreateVectorDb {
                                    result: true,
                                })),
                            })
                        }
                        _ => Err(RPCError {
                            rs_status_message: RPCErrors::InvalidParams,
                            rs_error_data: None,
                        }),
                    }
                }
                "UPSERT_VECTORS" => {
                    match rq_params {
                        RPCReqParams::GeneralReq {
                            method_params: Some(RPCReqMethodParams::UpsertVectors {
                                uv_vector_db_name,
                                uv_vector,
                            }),
                            ..
                        } => {
                            let vss = lookup_vector_store(&uv_vector_db_name).await;
                            match vss {
                                Some(vs) => {
                                    let levels = run_upload(&vs, &uv_vector).await;
                                    let stats = calculate_statistics(&[]); // Placeholder for levels
                                    Ok(RPCMessage::RPCResponse {
                                        rs_status_code: 200,
                                        pretty,
                                        rs_resp: Ok(Some(RPCResponseBody::RespUpsertVectors {
                                            insert_stats: stats,
                                        })),
                                    })
                                }
                                None => Ok(RPCMessage::RPCResponse {
                                    rs_status_code: 400,
                                    pretty,
                                    rs_resp: Err(RPCError {
                                        rs_status_message: RPCErrors::InvalidParams,
                                        rs_error_data: Some(format!(
                                            "Vector database '{}' does not exist",
                                            uv_vector_db_name
                                        )),
                                    }),
                                }),
                            }
                        }
                        _ => Err(RPCError {
                            rs_status_message: RPCErrors::InvalidParams,
                            rs_error_data: None,
                        }),
                    }
                }
                "VECTOR_KNN" => {
                    info!("KNN vectors");
                    match rq_params {
                        RPCReqParams::GeneralReq {
                            method_params: Some(RPCReqMethodParams::VectorKNN {
                                knn_vector_db_name,
                                knn_vector,
                            }),
                            ..
                        } => {
                            let vss = lookup_vector_store(&knn_vector_db_name).await;
                            match vss {
                                Some(vs) => {
                                    let knn = vector_knn(&vs, &knn_vector).await;
                                    Ok(RPCMessage::RPCResponse {
                                        rs_status_code: 200,
                                        pretty,
                                        rs_resp: Ok(Some(RPCResponseBody::RespVectorKNN { knn })),
                                    })
                                }
                                None => Ok(RPCMessage::RPCResponse {
                                    rs_status_code: 400,
                                    pretty,
                                    rs_resp: Err(RPCError {
                                        rs_status_message: RPCErrors::InvalidParams,
                                        rs_error_data: Some(format!(
                                            "Vector database '{}' does not exist",
                                            knn_vector_db_name
                                        )),
                                    }),
                                }),
                            }
                        }
                        _ => Err(RPCError {
                            rs_status_message: RPCErrors::InvalidParams,
                            rs_error_data: None,
                        }),
                    }
                }
                _ => Err(RPCError {
                    rs_status_message: RPCErrors::InvalidMethod,
                    rs_error_data: None,
                }),
            }
        }
        _ => Err(RPCError {
            rs_status_message: RPCErrors::InvalidRequest,
            rs_error_data: None,
        }),
    }
}

async fn get_db() -> Arc<Mutex<()>> {
    // Placeholder for getting DB connection
    Arc::new(Mutex::new(()))
}

async fn get_ain() -> Arc<Mutex<()>> {
    // Placeholder for getting AIN connection
    Arc::new(Mutex::new(()))
}

async fn lookup_user_data(session_key: &str) -> Option<(String, i32, i32, time::Instant, Vec<String>)> {
    // Placeholder for looking up user data
    None
}

async fn update_user_data(
    session_key: &str,
    name: String,
    quota: i32,
    used: i32,
    exp: time::Instant,
    roles: Vec<String>,
) {
    // Placeholder for updating user data
}

async fn delete_user_data(session_key: &str) {
    // Placeholder for deleting user data
}

async fn get_current_time() -> time::Instant {
    time::Instant::now()
}

async fn init_vector_store(name: &str, dim: usize, max: Option<f32>, min: Option<f32>) {
    // Placeholder for initializing vector store
}

async fn lookup_vector_store(name: &str) -> Option<Vec<f32>> {
    // Placeholder for looking up vector store
    None
}

async fn run_upload(vs: &Vec<f32>, vecs: &Vec<Vec<f32>>) -> Vec<i32> {
    // Placeholder for running upload
    vec![]
}

fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

async fn vector_knn(vs: &Vec<f32>, vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}

async fn login(user: &str, pass: &str) -> AuthResp {
    // Placeholder for login function
    AuthResp {
        session_key: None,
        calls_used: 0,
        calls_remaining: 0,
    }
}
