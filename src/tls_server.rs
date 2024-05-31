use rayon::ThreadPool;
use rustls_pemfile::Item;
use std::arch::x86_64;
use std::sync::{Arc, Mutex};
use crate::models::rpc::{

    RPCError, RPCErrors, RPCMessage, RPCReqMethodParams, RPCReqParams, RPCResponseBody,AuthResp,
};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use rustls::*;
use rustls::ServerConnection;
use rustls::StreamOwned;
use serde::{Deserialize, Serialize};
use log::{info, error};
use rayon::ThreadPoolBuilder;
use rustls::pki_types::{CertificateDer, CertificateRevocationListDer, PrivateKeyDer};
use std::{fs, net};

fn handle_rpc_request(msg: RPCMessage) -> RPCMessage {
    match msg {
        RPCMessage::RPCRequest { rq_method, rq_params } => {
            // Process the request and create a response
            RPCMessage::RPCResponse {
                rs_status_code: 200,
                pretty: false,
                rs_resp: Ok(Some(RPCResponseBody::AuthenticateResp {
                    auth: AuthResp {
                        session_key: Some("dummy_session_key".to_string()),
                        calls_used: 0,
                        calls_remaining: 10,
                    },
                })),
            }
        }
        _ => RPCMessage::RPCResponse {
            rs_status_code: 500,
            pretty: false,
            rs_resp: Err(RPCError {
                rs_status_message: "Invalid Request".to_string(),
                rs_error_data: None,
            }),
        },
    }
}

fn handle_client(mut stream: StreamOwned<ServerConnection, std::net::TcpStream>, tx: Arc<Mutex<Vec<RPCMessage>>>, pool: Arc<ThreadPool>) {
    let mut buffer = vec![0; 4096];

    loop {
        match stream.read(&mut buffer) {
            Ok(0) => return, // Connection closed
            Ok(n) => {
                let msg: RPCMessage = match serde_json::from_slice(&buffer[..n]) {
                    Ok(msg) => msg,
                    Err(e) => {
                        error!("Failed to deserialize message: {:?}", e);
                        continue;
                    }
                };

                let tx_clone = tx.clone();
                pool.spawn(move || {
                    let response = handle_rpc_request(msg);

                    let response_data = serde_json::to_vec(&response).unwrap();
                    let mut tx_lock = tx_clone.lock().unwrap();
                    tx_lock.push(response);

                    if let Err(e) = stream.write_all(&response_data) {
                        error!("Failed to write to socket: {:?}", e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to read from socket: {:?}", e);
                return;
            }
        }
    }
}

fn load_private_key(filename: &str) -> PrivateKeyDer<'static> {
    let keyfile = fs::File::open(filename).expect("cannot open private key file");
    let mut reader = BufReader::new(keyfile);

    loop {
        match rustls_pemfile::read_one(&mut reader).expect("cannot parse private key .pem file") {
            Some(rustls_pemfile::Item::Pkcs1Key(key)) => return key.into(),
            Some(rustls_pemfile::Item::Pkcs8Key(key)) => return key.into(),
            Some(rustls_pemfile::Item::Sec1Key(key)) => return key.into(),
            None => break,
            _ => {}
        }
    }

    panic!(
        "no keys found in {:?} (encrypted keys not supported)",
        filename
    );
}

fn load_certs(filename: &str) -> Vec<CertificateDer<'static>> {
    let certfile = fs::File::open(filename).expect("cannot open certificate file");
    let mut reader = BufReader::new(certfile);
    rustls_pemfile::certs(&mut reader)
        .map(|result| result.unwrap())
        .collect()
}

fn main() {

    let addr = "127.0.0.1:8443";
    let listener = TcpListener::bind(addr).expect("Failed to bind address");

    let certs = load_certs("certs/server.pem");
    let key = load_private_key("certs/server.key");

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .expect("bad certificate/key");

    let config = Arc::new(config);
    let tx = Arc::new(Mutex::new(Vec::new()));
    let pool = Arc::new(ThreadPoolBuilder::new().build().unwrap());

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let config = config.clone();
                let tx = tx.clone();
                let pool = pool.clone();

                thread::spawn(move || {
                    let session = ServerConnection::new(config);
                    let stream = StreamOwned::new(session, stream);
                    handle_client(stream, tx, pool);
                });
            }
            Err(e) => error!("Failed to accept connection: {:?}", e),
        }
    }
}
