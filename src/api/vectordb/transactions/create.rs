use crate::models::{types::get_app_env, versioning::VersionHasher};
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

    let mut cot_arc = vec_store.current_open_transaction.clone();

    if cot_arc.get().is_some() {
        return HttpResponse::Conflict()
            .body("Cannot create a new transaction while there's an on-going transaction");
    }

    let ver = vec_store.get_current_version().unwrap();
    let new_ver = ver.version + 1;
    let mut hasher = VersionHasher::new();
    let hash = hasher.generate_hash("main", new_ver, None, None);
    let transaction_id = hash.hash.clone();

    cot_arc.update(Some(hash));

    HttpResponse::Ok().json(json!({
        "transaction_id": transaction_id
    }))
}
