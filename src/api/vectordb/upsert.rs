use actix_web::{web, HttpResponse};

use crate::{
    api_service::run_upload,
    app_context::AppContext,
    convert_vectors,
    models::{
        rpc::{RPCResponseBody, UpsertVectors},
        types::get_app_env,
    },
};

// Route: `/vectordb/upsert`
pub(crate) async fn upsert(
    web::Json(body): web::Json<UpsertVectors>,
    ctx: web::Data<AppContext>,
) -> HttpResponse {
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
    }
    .clone();

    if vec_store.current_open_transaction.clone().get().is_some() {
        return HttpResponse::Conflict()
            .body("Cannot upsert while there's an on-going transaction");
    }

    // Call run_upload with the extracted parameters
    web::block(move || {
        // TODO: handle the error
        let _ = run_upload(ctx.into_inner(), vec_store, convert_vectors(body.vectors));
    })
    .await
    .unwrap();
    let response_data = RPCResponseBody::RespUpsertVectors { insert_stats: None };
    HttpResponse::Ok().json(response_data)
}
