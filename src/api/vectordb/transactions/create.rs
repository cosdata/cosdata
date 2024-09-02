use crate::models::types::get_app_env;
use actix_web::{web, HttpResponse};
use serde_json::json;

// Route: `/vectordb/{database_name}/transactions`
pub(crate) async fn create(database_name: web::Path<String>) -> HttpResponse {
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return HttpResponse::InternalServerError().body("Env initialization error"),
    };

    let Some(vec_store) = env.vector_store_map.get(&database_name.into_inner()) else {
        return HttpResponse::NotFound().body("Vector store not found");
    };

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();

    if current_open_transaction_arc.get().is_some() {
        return HttpResponse::Conflict()
            .body("Cannot create a new transaction while there's an on-going transaction");
    }

    let Ok(transaction_id) = vec_store.vcs.add_next_version("main") else {
        return HttpResponse::InternalServerError().body("LMDB error");
    };

    current_open_transaction_arc.update(Some(transaction_id));

    HttpResponse::Ok().json(json!({
        "transaction_id": *transaction_id
    }))
}
