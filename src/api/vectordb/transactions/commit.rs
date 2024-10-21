use crate::models::types::get_app_env;
use actix_web::{web, HttpResponse};

// Route: `/vectordb/{database_name}/transactions/{transaction_id}/commit`
pub(crate) async fn commit(path_data: web::Path<(String, u32)>) -> HttpResponse {
    let (database_name, transaction_id) = path_data.into_inner();
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return HttpResponse::InternalServerError().body("Env initialization error"),
    };
    let Some(vec_store) = env.vector_store_map.get(&database_name) else {
        return HttpResponse::NotFound().body("Vector store not found");
    };

    let mut current_open_transaction_arc = vec_store.current_open_transaction.clone();
    let current_open_transaction = current_open_transaction_arc.get();
    let Some(current_transaction_id) = current_open_transaction else {
        return HttpResponse::NotFound().body("Transaction not found");
    };

    if **current_transaction_id != transaction_id {
        return HttpResponse::NotFound().body("Transaction not found");
    }

    vec_store
        .current_version
        .clone()
        .update(current_transaction_id.clone());
    current_open_transaction_arc.update(None);

    HttpResponse::Ok().finish()
}
