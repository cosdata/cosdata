use crate::models::types::get_app_env;
use actix_web::{web, HttpResponse};

// Route: `/vectordb/{database_name}/transactions/{transaction_id}/update`
pub(crate) async fn update(path_data: web::Path<(String, String)>) -> HttpResponse {
    let (database_name, transaction_id) = path_data.into_inner();
    let env = match get_app_env() {
        Ok(env) => env,
        Err(_) => return HttpResponse::InternalServerError().body("Env initialization error"),
    };
    let Some(vec_store) = env.vector_store_map.get(&database_name) else {
        return HttpResponse::NotFound().body("Vector store not found");
    };

    todo!()
}
