use actix_web::{web, HttpResponse};

use crate::{
    api_service::run_upload,
    convert_vectors,
    models::{
        rpc::{RPCResponseBody, UpsertVectors},
        types::get_app_env,
    },
};

// Route: `/vectordb/upsert`
pub(crate) async fn upsert(web::Json(body): web::Json<UpsertVectors>) -> HttpResponse {
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return HttpResponse::InternalServerError().body("Env initialization error"),
    };
    // Try to get the vector store from the environment
    let vec_store = match env.vector_store_map.get(&body.vector_db_name) {
        Some(store) => store,
        None => {
            // Vector store not found, return an error response
            return HttpResponse::InternalServerError().body("Vector store not found");
        }
    };

    if vec_store.current_open_transaction.read().unwrap().is_some() {
        return HttpResponse::Conflict()
            .body("Cannot upsert while there's an on-going transaction");
    }

    // Call run_upload with the extracted parameters
    let __result = run_upload(vec_store.clone(), convert_vectors(body.vectors)).await;

    let response_data = RPCResponseBody::RespUpsertVectors { insert_stats: None }; //
    HttpResponse::Ok().json(response_data)
}
