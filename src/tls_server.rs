use rayon::ThreadPool;

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

fn handle_client(mut stream: StreamOwned<ServerSession, std::net::TcpStream>, tx: Arc<Mutex<Vec<RPCMessage>>>, pool: Arc<ThreadPool>) {
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

fn main() {
    env_logger::init();

    let addr = "127.0.0.1:8443";
    let listener = TcpListener::bind(addr).expect("Failed to bind address");

    let certs = load_certs("certs/server.pem");
    let key = load_private_key("certs/server.key");

    let config = ServerConfig::builder()
        .with_safe_defaults()
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
                    let session = ServerSession::new(&config);
                    let stream = StreamOwned::new(session, stream);
                    handle_client(stream, tx, pool);
                });
            }
            Err(e) => error!("Failed to accept connection: {:?}", e),
        }
    }
}
