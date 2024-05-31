
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};
use log::info;
use crate::models::encoding_format::EncodingFormat;
use crate::models::rpc::{
    RPCError, RPCErrors, RPCMessage, RPCReqMethodParams, RPCReqParams, RPCResponseBody,
};
use crate::{models::user::{AuthResp, Statistics}};
use chrono::prelude::*;




fn auth_login_client(
    rq_method: String,
    username: String,
    password: String,
    pretty: bool,
) -> Result<RPCMessage, RPCError> {
    let dbe = get_db();

    if rq_method == "AUTHENTICATE" {
        let resp = login(&username, &password);
        Ok(RPCMessage::RPCResponse {
            rs_status_code: 200,
            pretty,
            rs_resp: Ok(Some(RPCResponseBody::AuthenticateResp { auth: resp })),
        })
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

fn delegate_request(enc_req: RPCMessage) -> Result<RPCMessage, RPCError> {
    let dbe = get_db();
    let ain = get_ain();
  

    match enc_req {
        RPCMessage::RPCRequest { rq_method, rq_params }  => {
        let rqm = rq_method;
        
        match rq_params {
            RPCReqParams::AuthenticateReq {
                username,
                password,
                pretty_print,
            } => auth_login_client(rqm, username, password, pretty_print),
            RPCReqParams::GeneralReq {
                session_key,
                pretty_print,
                method_params } => {

                let rq_method_param = method_params;
                let user_data = lookup_user_data(&session_key);

                match user_data {
                    None => go_get_resource(rqm,rq_method_param, vec![], &session_key, &pretty_print),
                    Some((name, quota, used, exp, roles)) => {
                        let curtm = Utc::now().to_rfc3339();
                        if exp > curtm && quota > used {
                            if (used + 1) % 100 == 0 {
                                // Placeholder for additional logic
                            }
                            update_user_data(&session_key, name, quota, used + 1, exp, &roles);
                            go_get_resource(rqm,rq_method_param, roles, &session_key, &pretty_print)
                        } else {
                            delete_user_data(&session_key);
                            Ok(RPCMessage::RPCResponse {
                                rs_status_code: 200,
                                pretty: true,
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
        }},
        _ => Err(RPCError {
            rs_status_message: RPCErrors::InvalidRequest,
            rs_error_data: None,
        }),
    }
}

fn go_get_resource(
    rq_method: String,
    rq_method_params: Option<RPCReqMethodParams> ,
    roles: Vec<String>,
    sess_key: &str,
    pretty:&bool,
) -> Result<RPCMessage, RPCError> {
    let dbe = get_db();
    let ain = get_ain();


            match rq_method.as_str() {
                "ADD_USER" => {
                    // Placeholder for ADD_USER logic
                    unimplemented!()
                }
                "CREATE_VECTOR_DB" => {
                    info!("Creating Vector DB");
                    match  rq_method_params {
                            
                            Some(RPCReqMethodParams::CreateVectorDb {
                                    cv_vector_db_name,
                                    cv_dimensions,
                                    cv_max_val,
                                    cv_min_val,
                                }) => {
                            info!("Creating Vector DB: {}", cv_vector_db_name);
                            init_vector_store(
                                &cv_vector_db_name,
                                cv_dimensions as usize,
                                cv_max_val,
                                cv_min_val,
                            );
                            Ok(RPCMessage::RPCResponse {
                                rs_status_code: 200,
                                pretty: *pretty,
                                rs_resp: Ok(Some(RPCResponseBody::RespCreateVectorDb {
                                    result: true,
                                })),
                            })
                        }
                        __ => {Ok(RPCMessage::RPCResponse {
                            rs_status_code: 400,
                            pretty: *pretty,
                            rs_resp: Err(RPCError {
                                rs_status_message: RPCErrors::InvalidParams,
                                rs_error_data: Some( "Vector database exists".to_string() ),
                            }),
                        })}
                       
                    }}
                
                "UPSERT_VECTORS" => {
                    match rq_method_params {

                                Some(RPCReqMethodParams::UpsertVectors {
                                    uv_vector_db_name,
                                    uv_vector,
                                })  => {
                            let vss = lookup_vector_store(&uv_vector_db_name);
                            match vss {
                                Some(vs) => {
                                    let levels = run_upload(&vs, &uv_vector);
                                    let stats = calculate_statistics(&[]); // Placeholder for levels
                                    Ok(RPCMessage::RPCResponse {
                                        rs_status_code: 200,
                                        pretty: *pretty,
                                        rs_resp: Ok(Some(RPCResponseBody::RespUpsertVectors {
                                            insert_stats: stats,
                                        })),
                                    })
                                }
                                None => Ok(RPCMessage::RPCResponse {
                                    rs_status_code: 400,
                                    pretty: *pretty,
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
                    match  rq_method_params {
                                Some(RPCReqMethodParams::VectorKNN {
                                    knn_vector_db_name,
                                    knn_vector,
                                }) => {
                            let vss = lookup_vector_store(&knn_vector_db_name);
                            match vss {
                                Some(vs) => {
                                    let knn = vector_knn(&vs, &knn_vector);
                                    Ok(RPCMessage::RPCResponse {
                                        rs_status_code: 200,
                                        pretty: *pretty,
                                        rs_resp: Ok(Some(RPCResponseBody::RespVectorKNN { knn })),
                                    })
                                }
                                None => Ok(RPCMessage::RPCResponse {
                                    rs_status_code: 400,
                                    pretty: *pretty,
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


fn get_db() -> Arc<Mutex<()>> {
    // Placeholder for getting DB connection
    Arc::new(Mutex::new(()))
}

fn get_ain() -> Arc<Mutex<()>> {
    // Placeholder for getting AIN connection
    Arc::new(Mutex::new(()))
}

fn lookup_user_data(session_key: &str) -> Option<(String, i32, i32, String, Vec<String>)> {
    // Placeholder for looking up user data
    None
}

fn update_user_data(
    session_key: &str,
    name: String,
    quota: i32,
    used: i32,
    exp: String,
    roles: &Vec<String>,
) {
    // Placeholder for updating user data
}

fn delete_user_data(session_key: &str) {
    // Placeholder for deleting user data
}

fn init_vector_store(name: &str, dim: usize, max: Option<f32>, min: Option<f32>) {
    // Placeholder for initializing vector store
}

fn lookup_vector_store(name: &str) -> Option<Vec<f32>> {
    // Placeholder for looking up vector store
    None
}

fn run_upload(vs: &Vec<f32>, vecs: &Vec<Vec<f32>>) -> Vec<i32> {
    // Placeholder for running upload
    vec![]
}

fn calculate_statistics(_: &[i32]) -> Option<Statistics> {
    // Placeholder for calculating statistics
    None
}

fn vector_knn(vs: &Vec<f32>, vecs: &Vec<f32>) -> Vec<(i8, i8, String, f64)> {
    // Placeholder for vector KNN
    vec![]
}

fn login(user: &str, pass: &str) -> AuthResp {
    // Placeholder for login function
    AuthResp {
        session_key: None,
        calls_used: 0,
        calls_remaining: 0,
    }
}
